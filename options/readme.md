isTrain flag is not used anymore
TestOption is also not used

need to better organise options

needed functions
1. required/optional configurations
2. TODO: dynamic options? if we add a certain tag, we design the datasets? e.g. give the dataset file path, build the loader and call it?
3. classify the options based on their function/usage
    - timepoint control, e.g. when to save models, print loss and so on
    - variable construction, e.g. adding new parts to be saved
    - logic define, e.g. adding new process/function, the sw-x, switch on/off
4. when we have long complex options, how to check if they are valid? 
    - what if required options are not provided? and we parse the options wrong?
5. naming of the options, so that during coding, you know the option's nature from its name

# ideas

use name tag to organise the flow

how to trade-off between generality and customise?

# reasonings

options are important,
because it means how flexible we can be

# examples

## wrong config

an extra space
~~~
c
''
config_list
['i1-o32-t0-k5-p2-s2-fReLU', 'i32-o64-t0-k5-p2-s2-fReLU', 'i64-o128-t0-k5-p2-s2-fReLU', 'i128-o256-t0-k5-p2-s2-fReLU', '', 'i256-o256-t0-k5-p2-s2-fReLU']
~~~