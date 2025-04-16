import numpy as np
import torch
#from .ILD_alpha import alpha_to_LR
from slab import HRTF, Sound, Filter, Binaural
from collections import namedtuple

class kemar_env():
    """
    has to be used by single process only
    because otherwise the state is not shared

    acoustic environment for the RL agent


    init: specified frequency bands.
    input: angles
    output: frequency broadband sensory input, includeing ILD or spectrum
    function: 
        0. use white noise
        1. given angle, get the sensory input
        2. calculate the next angle based on action and init sound source position

    future improvement:
        1. computing faster ILD   
        2. add room simulation if needed

    comments: 
        you get the same problem for other RL environment simulations: not differentiable, costly to simulate 
        
    """
        # -=============== for RL
    def __init__(self, sigma=40, success_reward=100, step_punish=-0.1, ratio_coeff=[1.0, 1.0], fluc_level=[0, 0, 0 ,0], fluc_elev=[0, 0], sound_lib=None, filter_banks_n=None, sw_elev=False) -> None:
        """

        sigma: controls the hardness/precision of the environment

        refactor:
            we can make this a state-less class
            where we remove the inner state of s_angle, but pass it in everytime. Perhaps better for parallelisation

        ratio_coeff: the ratio of the left and right sound levels, for the ILD calculation [left, right]
        """
        #sigma = 20.0 # 5.0 # 20.0 # 20.0 # 5.0 # 1.0 # harder environment, but less noise
        #sigma = 40.0
        self.sigma = sigma # the gaussian-bell shpae reward curve sigma

        #accept_range = 5
        self.step_punish = step_punish # should not be too high or too low
        self.success_reward = success_reward #100.0 

        self.hrtf = HRTF.kemar() # sample rate is fixed to 44100.0
        if filter_banks_n is not None:
            # this is actually used as default. Because our option always give the filter_banks_n=24
            # align with the paper: Deep neural network models of sound localization reveal how perception is adapted to real-world environments 
            #  Filter centre frequencies ranged from 45 to 16,975 Hz
            # cos_filterbank(length=5000, bandwidth=0.3333333333333333, low_cutoff=0, high_cutoff=None, pass_bands=False, n_filters=None, samplerate=None)
            self.fbank = Filter.cos_filterbank(samplerate=self.hrtf.samplerate, pass_bands=True, n_filters=filter_banks_n, low_cutoff=20, high_cutoff=20000)
        else:
            self.fbank = Filter.cos_filterbank(samplerate=self.hrtf.samplerate, pass_bands=True)
        
        self.human_bandpass = Filter.band(kind='bp', frequency=(20, 20000), samplerate=self.hrtf.samplerate) # the default sample rate is different.

        self.freqs = self.fbank.filter_bank_center_freqs()
        self.ratio_coeff = ratio_coeff
        self.fluc_level = fluc_level
        self.fluc_elev=fluc_elev

        # assert self.fluc_level == 0 # otherwise, scaling does not make any sense

        self.sw_elev=sw_elev # switch for using 3D angle or not

        if sound_lib is None:
            self.sound_lib=None
        else:        
            # load the wav file
            self.sound_lib=Sound(sound_lib)
            self.sound_lib_len=self.sound_lib.data.shape[0]
            self.sound_lib_sr=self.sound_lib.samplerate

        # assert self.fluc_level >= 0 and self.fluc_level<1, "fluc_level should be [0, 1]"
        #self.sample_angle_a_sound()

        return

    def sample_angle_a_sound(self, angle=None, elev=None):
        """
            sample angle and sound
            similar to the reset() method

            sample a random init angle for the sound source
            sample a random sound
            called every episode

            TODO
                1. random sound levels?
                2. do not touch the angle for now 
                3. written in pytorch?
        """
        if angle is None:
            # sample
            # self.s_angle = np.random.uniform(-90, 90)        
            # self.s_angle = np.random.uniform(-120, 120)        
            # self.s_angle = np.random.uniform(-100, 100)        
            
            x = np.random.beta(0.2, 0.2) # [0, 1]
            self.s_angle = x*180-90 # [-90, 90]
        else:
            # use the input angle
            self.s_angle = angle

        if self.sw_elev:
            self.s_elev = elev
        else:
            pass # do nothing


        #self.sound = Sound.whitenoise(duration=1., samplerate=44100)
        if self.sound_lib is None:
            self.sound = Sound.whitenoise(duration=.1, samplerate=44100) # make sound shorter for faster calculation, 0.1s, 100ms
        else:
            # leave the 10*0.2s end not used
            sample_start=np.random.randint(0, self.sound_lib_len-10*int(0.2*self.sound_lib_sr))
            self.sound = Sound(self.sound_lib.data[sample_start:sample_start+int(0.2*self.sound_lib_sr)], samplerate=self.sound_lib_sr)

        return self.scale_angle_render_sound()

    #def reformat_state(self):
    def scale_angle_render_sound(self, angle=None, elev=None):
        """
            TODO how to do refactor correctly with the idea of compatability of previous models?
            TODO shall we separate the sound as well? This seems to be a pattern, do not mix the object states with the transfer functions
            same for the level fluc and so on
            make this rendering function independent from the object states
            so that we can do stateless rendering if needed somewhere else

            ---------

            only observe, no update on the status.

            scale the angle state from [-90, 90] to [0, 1]
            render the LR sound for the current angle

            should be called inside transit? not really, transit function handles rewards and done signals only

            NOTES:
            0. there is no batch operation yet 
            1. kemar use a degree based uni-system for angle, 
                see https://github.com/DrMarc/slab/blob/master/slab/hrtf.py#L302

        """
        with torch.no_grad():
            # TODO refactor this function, do not need to mix the scaling with the rendering
            self.scaled_angle = (self.s_angle+180)/360 # [-90, 90] will be [0.25, 0.75], this is useful to numerically normalize the output of the network

            if angle is None:
                angle = self.s_angle
            else:
                pass # we get the angle passed in
            scaled_angle = (angle+180)/360
            
            # solved-- the slab library use the same -1! why hrtf use a different coordinate system? need to change the sign of the angle
            # at https://github.com/DrMarc/slab/blob/c415cad74e9b5c5ed710542b36b3708f1b0519d1/slab/binaural.py#L270C1-L270C116 azimuth_to_ild()

            if self.sw_elev:
                # control the elevation angle
                if elev is None:
                    elev = self.s_elev # in degree, -90 to 90
                else: 
                    pass # we get the elev passed in
            else:
                # random sampling, do not control the elevation
                if self.fluc_elev[0] == self.fluc_elev[1]:
                    elev=self.fluc_elev[0]
                else:
                    elev=np.random.uniform(self.fluc_elev[0], self.fluc_elev[1])

            # interpolation with triangle method, instead of the nearest
            f_theta = self.hrtf.interpolate(-1*angle, elev, 'triangle') # 0 for 0 degree elevation 

            fn = Binaural(f_theta.apply(self.sound))

            # NOTE: this is a backup plan, not fully implemented yet. 20250408
            # apply the bandpass filter for human hearing range, before calculating the level
            # this is not a filter bank, just a single band pass filter to measure the sound.
            # TODO: here can add a Relu to make sure the level is positive, because it is calculated by human lower bound.
            total_left_level=self.human_bandpass.apply(fn.left).level
            total_right_level=self.human_bandpass.apply(fn.right).level

            noise_bank_left = self.fbank.apply(fn.left)
            noise_bank_right = self.fbank.apply(fn.right)
            #level_diffs = noise_bank_right.level - noise_bank_left.level

            scaled_angle_rendered_binaual = namedtuple('sarb','angle_01 l_levels r_levels angle_elev_deg total_left_level total_right_level')

            # #r_input, l_input = alpha_to_LR(self.s_angle.cpu().detach().numpy())
            # r_input, l_input = alpha_to_LR(self.s_angle)
            # LR = torch.tensor([r_input, l_input], dtype=torch.float) # reuse the name s_ILD, but for batch=1 only

            # l_levels = noise_bank_left.level * self.ratio_coeff[0]
            # r_levels = noise_bank_right.level * self.ratio_coeff[1]

            l_levels = noise_bank_left.level
            r_levels = noise_bank_right.level

            # get only the non-negative values for the ndarray

            l_levels = np.maximum(l_levels, 0)
            r_levels = np.maximum(r_levels, 0)
            
            # update the ratio coeff
            l_levels = l_levels * self.ratio_coeff[0]
            r_levels = r_levels * self.ratio_coeff[1]

            # this does not have good meaning, sound level change should be added not multiplied.
            # if self.fluc_level != 0:
            #     # sample uniform between 1+- fluc_level for the sound levels
            #     k=np.random.uniform(1-self.fluc_level, 1+self.fluc_level)
            #     l_levels = l_levels*k
            #     r_levels = r_levels*k
            

            ## these are the db changes
            ## sound level may random sample

            # sample uniform between 1+- fluc_level for the sound levels
            kl=np.random.uniform(self.fluc_level[0], +self.fluc_level[1]) # default the fluc_level[0] == fluc_level[1] = 0
            l_levels = l_levels+kl

            kr=np.random.uniform(self.fluc_level[2], +self.fluc_level[3]) # default the fluc_level[0] == fluc_level[1] = 0
            r_levels = r_levels+kr

        # return scaled_angle, LR
        return scaled_angle_rendered_binaual(scaled_angle, l_levels, r_levels, elev, total_left_level, total_right_level)
    
    # def transit360_wo_reward(self, action, sw_render=True):
    #     """
    #         only transit the states
    #         do not provide any external reward

    #         this works for [0, 360] angle system
    #         true
    #     """
    #     with torch.no_grad():
    #         state0 = self.s_angle
            
    #         # new angle need to be circular, rather than simple linear subtraction?
    #         angle = state0 - (action.cpu().detach().numpy()*360)
    #         state1 = angle
    #         self.s_angle = state1
        
    #     if sw_render:
    #         return self.scale_angle_render_sound()
    #     else:
    #         return None

    def transit_wo_reward(self, action, sw_render=True):
        """
            only transit the states
            do not provide any external reward


            this only works for [-90, 90] angle system?
            true
        """
        with torch.no_grad():
            state0 = self.s_angle
            angle = state0 - action # env_action=(action[0,0].cpu().detach().numpy()*360-180)*scale_factor # sigmoid output is in [1/8, 3/8]
            state1 = angle
            self.s_angle = state1
        
        if sw_render:
            return self.scale_angle_render_sound()
        else:
            return None

    def stateless_render(self, action):
        """
            do not change the state

            agent facing 0, 
            agent hear sound from s_angle
            anent take action to rotate head, but turn back to 0 after action, so the s_angle does not change
        """
        with torch.no_grad():
            state0 = self.s_angle
            angle = state0 - (action.cpu().detach().numpy()*360-180)
            state1 = angle
            self.s_angle = state1 # so we can render the sound

            # this is wrong!
            # self.s_angle = action.cpu().detach().numpy()*360-180
        
            sarb=self.scale_angle_render_sound()
            self.s_angle=state0 # restore
            return sarb

    def stateless_render_3D(self, angle, elev):
        """
            NOTE avoid using this funcition
            we move the status transition out to the main procedure 
            directly call the rendering function with the given angle and elev

            ---------

            angle is the horizontal angle, in degree
            elev is the vertical angle, in degree

            elev should be within [-90, 90]

            do not change the state

            agent facing 0, 
            agent hear sound from s_angle
            anent take action to rotate head, but turn back to 0 after action, so the s_angle does not change
        """
        with torch.no_grad():


            new_angle = self.s_angle - angle # s_angle and angle are both in degree

            # 90 degree is the top, -90 is the bottom
            # new_elev = self.s_elev - elev # s_elev and elev are both in degree
            # but the result may be out of range, if it is 90-(-30)? 120, new elev should be 120-90=30, but at the back, that changes the horizontal angle as well? YES. Tricky
            # Avoid this by limiting the rotation range? -- OK
            # NOTE: here we use a simple calculation, it may cause problems for the full range, but OK if we limit the range 
            new_elev = self.s_elev - elev
            
            if new_elev > 90:
                new_elev = 90 - (new_elev - 90) # 90 - (120-90) = 60
                new_angle = new_angle + 180 # turn back
            elif new_elev < -90:
                new_elev = -90 - (new_elev + 90) # -90 - 30 = -120, -90 - (-120 + 90) = -90 + 30 = -60
                new_angle = new_angle + 180 

            sarb=self.scale_angle_render_sound(angle=new_angle, elev=new_elev)
            return sarb


    def transit_w_reward(self, action):
        """
            a very simple environment 

            this need to be changed if we want to use the inner reward rather than the external reward

            inner state:
                angle of the sound source position, no control over the sound level or frequence
            output:
                done:
                    -1 if out-of-bound 
                    0 if no reward out
                    1 if get the reward
            input: 
                action:
                    rotation of head, in scaled angle
        """
        # TODO: re-write this function properly

        with torch.no_grad():

            state0 = self.s_angle

            angle = state0 - (action.cpu().detach().numpy()*360-180)
            state1 = angle
            self.s_angle = state1


            # Default values
            done = 0
            reward = self.step_punish # default punishing

            if angle > 90 or angle < -90:
                done = -1
                #reward += -1.0 # punish for out of bound
            else:
                #m = torch.distributions.normal.Normal(torch.tensor([0.0]).to(device=angle.device), torch.tensor([sigma]).to(device=angle.device)) # make distribution m in GPU
                # NOTE angle need to be in degree units
                m = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([self.sigma])) # make distribution m in GPU
                accept_rate = 2*(1.0-m.cdf(torch.tensor(np.abs(angle))))
                # sample from bernuli
                sb = torch.bernoulli(accept_rate)

                # --------- random feedback with equal reward, realistic but harder
                if sb>0:
                   #reward += 100.0 # the value matters? 
                   reward += self.success_reward # 100.0 # the value matters? 
                   done = 1

                # # ---------- deterministic feedback with expectation
                # #if sb>0:
                # if 1:
                #     done = 1
                #     reward += 100.0*accept_rate.item() # using expectation reduce the variance, should help with learning 
                #     #reward += 100.0 # higer reward makes learning easier, because the weight of the gradient is higher
                # #reward = step_punish + l # instead of sampling the Bernulli according to the l probability, we just use the expectation, make things easier

            # self.s_angle = state1
        # ?need to return the new status as well self.scale_angle_render_sound()?
        # NO need.
        return reward, done
        # pass