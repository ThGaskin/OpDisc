import ruamel.yaml as yaml
import os

to_plot = [(5, 0.6), (5, 0.95), (7, 0.1), (7, 0.3), (7, 0.6), (7, 0.95), (9, 0.1), (9, 0.3), (9, 0.6), (9, 0.95)]

with open('OpDisc_plots_backup.yml') as file:
    A = yaml.safe_load(file)

for item in to_plot:
    
    A['area']['select']['subspace']['number_of_groups']=item[0]
    A['area']['select']['subspace']['homophily_parameter']=item[1]

    with open(r'OpDisc_plots.yml', 'w') as file:
        documents = yaml.dump(A, file)
   
    os.system('utopia eval OpDisc /Users/thomasgaskin/utopia_output/OpDisc/201009-143102 --plot-only area')
        
