# %* *****************************************************************************
# %  *  Name:   Mini Thomas
# %  *
# %  *  Title:  Trust Inference using Bayesian Network
# %  *  Description:
# %  *  This program computes the probability of trust using Bayesian Network(BN).
# In this program the bnlearn library is used. The steps followed are:
# - DAG: A DAG is estimated to capture the dependencies between the variables.
# - Parameter learning: Based on the expert knowledge data (discrete probabilities) and the DAG,
#    estimate the conditional probability distributions of the individual variables.
# - Inference: Given the learned model determine the exact probability values for our queries.
# %  *
# %  *  Written:       1September2021
# %  *  Last updated:  5October2021
# %  *
# %  **************************************************************************** */


import bnlearn as bn
# Import the library
from pgmpy.factors.discrete import TabularCPD

import networkx as nx
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------------------------------#
# Firstly, Define the structure- the causal dependencies of the system.
# x1; x2; x3; x4 represent energy, memory, RSSI, latency
# x5; x6; x7; x8 represent confidentiality, integrity, compliance,threat safe;
# x9; x10; x11; x12 represent data storage, data usage,transparent, anonymize;
# x13; x14 represent performance,reliability;
# x15; x16 represent operations, standards;
# x17; x18 represent data reuse, data protection;
# x19; x20; x21 represent robustness, security, privacy, and x22 represent trust.
# ------------------------------------------------------------------------------------------------------#

dag= nx.DiGraph()
map(dag.add_node,range(6))
pos={0:(-5,0),1:(-4,0),2:(-3,0),3:(-2,0),4:(-1,0),5:(0,0),6:(1,0),7:(2,0),8:(3,0),9:(4,0),10:(5,0),11:(6,0),
     12:(-4,-0.2),13:(-2.5,-0.2),14:(-0.5,-0.2),15:(1.5,-0.2),16:(3,-0.2),17:(4,-0.2),
     18:(-1.5,-0.6), 19:(0,-0.6),20:(1.5,-0.6),21:(0,-1)
     }
nx.draw(dag, pos)
nx.draw_networkx_edges(dag,pos,
edgelist=[(0,12),(1,12),(1,13),(2, 13),(3,13),
(4,13),(4,14),(4,14),(5, 15),(6, 15),(7,15),
(8,15),(8,16),(9,16),(10,17),(11,17),
(12,18),(13,18),(13,19),(14,19),(15,19),
(16,20),(17,20),(18,21) ,(19,21),(20,21)],edge_color='r')


nx.draw_networkx_nodes(dag,pos, nodelist=[0,1,2,3,4,5,6,7,8,9,10,11],
node_color='y',
node_size=500,
alpha=0.8)
nx.draw_networkx_nodes(dag,pos, nodelist=[12,13,14,15,16,17,18,19,20,21],
node_color='b',
node_size=500,
alpha=0.8)
labels={}
labels[0]=r'$x1$'
labels[1]=r'$x2$'
labels[2]=r'$x3$'
labels[3]=r'$x4$'
labels[4]=r'$x5$'
labels[5]=r'$x6$'
labels[6]=r'$x7$'
labels[7]=r'$x8$'
labels[8]=r'$x9$'
labels[9]=r'$x10$'
labels[10]=r'$x11$'
labels[11]=r'$x12$'
labels[12]=r'$x13$'
labels[13]=r'$x14$'
labels[14]=r'$x15$'
labels[15]=r'$x16$'
labels[16]=r'$x17$'
labels[17]=r'$x18$'
labels[18]=r'$x19$'
labels[19]=r'$x20$'
labels[20]=r'$x21$'
labels[21]=r'$x22$'
nx.draw_networkx_labels(dag, pos, labels, font_size=12)
plt.show()


# Define the causal dependencies based on your expert/domain knowledge.
# Left is the source, and right is the target node.
edges = [('memory', 'performance'),
         ('power', 'performance'),
         # ('memory', 'reliability'),
         ('rssi', 'reliability'),
         ('latency', 'reliability'),
         ('confidentiality', 'operations'),
         ('integrity', 'operations'),
         ('compliance', 'standards'),
         ('threatsafe', 'standards'),
       #('datastorage', 'reuse'),
       # ('datausage', 'reuse'),
       ('transparent', 'privacy'),
        # ('transparent', 'protection'),
       #('anonymize', 'protection'),
         ('performance', 'robustness'),
         ('reliability', 'robustness'),
         ('operations', 'security'),
         ('standards', 'security'),
        # ('reuse', 'privacy'),
        #('protection', 'privacy'),
         ('robustness', 'trust1'),
         ('security', 'trust1'),
        ('trust1', 'trust'),
        ('privacy', 'trust')]

# Create the DAG
DAG = bn.make_DAG(edges)

# Plot the DAG (static)
#bn.plot(DAG)

# Plot the DAG (interactive)
#bn.plot(DAG, interactive=True)

# DAG is stored in an adjacency matrix
DAG["adjmat"]

# ------------------------------------------------------------------------------------------------------#
# Next, we start building the probability table for non-descendant Nodes
# (i.e. the nodes that have no parents) and no dependencies
# Note:we consider - a High state of (1)= good for system trust, Low(0)=bad
# ------------------------------------------------------------------------------------------------------#

# Memory Node
# The memory node has two states (high or low (consumption)) and no dependencies. Calculating the probability
# from an expert view - witnessed 70% of the time memory consumption is high.
# As the probabilities should add up to 1, low consumption should be 30% of the time.
# The CPT for memory looks as following:
cpt_memory = TabularCPD(variable='memory', variable_card=2, values=[[0.7], [0.3]])
print(cpt_memory)

# Power Node
# The power node has two states (high or low (consumption)). Calculating the probability
# from an expert view - witnessed 60% of the time power consumption is high.
# As the probabilities should add up to 1, low consumption should be 40% of the time.
cpt_power = TabularCPD(variable='power', variable_card=2, values=[[0.6], [0.4]])
print(cpt_power)

# RSSI Node
# The rssi node has two states (high or low (strength)). Calculating the probability
# from an expert view - witnessed 50% of the time signal strength is high.
# As the probabilities should add up to 1, low consumption should be 50% of the time.
cpt_rssi = TabularCPD(variable='rssi', variable_card=2, values=[[0.5], [0.5]])
print(cpt_rssi)

# Latency Node
# The latency node has two states (high or low (strength)). Calculating the probability
# from an expert view - witnessed 70% of the time signal latency is low .
# As the probabilities should add up to 1, low consumption should be 30% of the time the latency is high.
cpt_latency = TabularCPD(variable='latency', variable_card=2, values=[[0.3], [0.7]])
print(cpt_latency)

# Confidentiality Node
# The confidentiality node has two states (high or low (encryption)). Calculating the probability
# from an expert view - witnessed 60% of the time the system has high encryption and 40% of the time conf. is low.
cpt_confidentiality = TabularCPD(variable='confidentiality', variable_card=2, values=[[0.4], [0.6]])
print(cpt_confidentiality)

# Integrity Node
# The integrity node has two states (high or low (high means high chance that no outsider can change sensor data))
# Calculating the probability from an expert view - witnessed 80% of the time
# the system has high encryption and 20% of the time integrity. is low.
cpt_integrity = TabularCPD(variable='integrity', variable_card=2, values=[[0.2], [0.8]])
print(cpt_integrity)

# Compliance Node
# The Compliance node has two states (high or low (high means high chance that system is compliant with security
# standards & protocols). Calculating the probability from an expert view - witnessed 90% of the time
# the system is compliant and 10% of the time compliance. is low.
cpt_compliance = TabularCPD(variable='compliance', variable_card=2, values=[[0.1], [0.9]])
print(cpt_compliance)

# Threatsafe Node
# The threatsafe node has two states (high or low (high means high chance that no threat happens/system is safe.
# Calculating the probability from an expert view - witnessed 60% of the time
# the system is safe and 40% of the time integrity. is low.
cpt_threatsafe = TabularCPD(variable='threatsafe', variable_card=2, values=[[0.4], [0.6]])
print(cpt_threatsafe)


# # Datastorage Node
# # The datastorage node has two states (high or low (high means high chance that data stored is safe.
# # Calculating the probability from an expert view - witnessed 60% of the time
# # the system is safe and 40% of the time integrity. is low.
# cpt_datastorage = TabularCPD(variable='datastorage', variable_card=2, values=[[0.4], [0.6]])
# print(cpt_datastorage)
#
#
# # DataUsage Node
# # The datausage node has two states (high or low (high means high chance that data is only used for the
# # specific application and not used for unrelated purpose.
# # Calculating the probability from an expert view - witnessed 80% of the time
# # the data is not used for unethical purpose while 20% of the time is use was not clear.
# cpt_datausage = TabularCPD(variable='datausage', variable_card=2, values=[[0.2], [0.8]])
# print(cpt_datausage)


# Transparent Node
# The Transparent node has two states (high or low (high means the data flow is transparent to the
# patient and primary SH.Calculating the probability from an expert view - witnessed 60% of the time
# the system is safe and 40% of the time data flow is not transparent.
cpt_transparent = TabularCPD(variable='transparent', variable_card=2, values=[[0.4], [0.6]])
print(cpt_transparent)


# Anonymize Node
# The Anonymize node has two states (high or low (high means high chance that patients private data is anonymized .
# Calculating the probability from an expert view - witnessed 40% of the time
# the data was strongly anonymized and 60% of the time not.
# cpt_anonymize = TabularCPD(variable='anonymize', variable_card=2, values=[[0.6], [0.4]])
# print(cpt_anonymize)


# ------------------------------------------------------------------------------------------------------#
# Secondly, we start building the probability table for the Indicator- descendant nodes
# (i.e. the nodes that have parents) based on conditional probability
# ------------------------------------------------------------------------------------------------------#

# Performance Node
# The performance  node has two states and is conditioned by two-parent nodes; memory and power.
# Here we define the probability of performance given the state of memory and power.
# In total, we have to specify 8 conditional probabilities (2 states ^ 3 nodes).
cpt_performance = TabularCPD(variable='performance', variable_card=2,
                             values=[[0.9, 0.6, 0.6, 0.1],
                                     [0.1, 0.4, 0.4, 0.9]],
                             evidence=['memory', 'power'],
                             evidence_card=[2, 2])
print(cpt_performance)


# Reliability Node
# The reliability node has two states and is conditioned by three-parent nodes; memory,latency and rssi.
# Here we define the probability of performance given the state of memory,latency and rssi.
# In total, we have to specify 16 conditional probabilities (2 states ^ 4 nodes).
# 3 evidence is still showing error , so for now using 2 evidence, as shown below
# cpt_reliability = TabularCPD(variable='reliability', variable_card=2,
#                       values=[[0.3,0.1,0.7,0.4,0.6,0.4,0.99,0.7],
#                               [0.7,0.9,0.3,0.6,0.4,0.6,0.01,0.3]],
#                       evidence=['memory','latency', 'rssi'], evidence_card=[3,2])
cpt_reliability = TabularCPD(variable='reliability', variable_card=2,
                             values=[[0.9, 0.6, 0.6, 0.1],
                                     [0.1, 0.4, 0.4, 0.9]],
                             evidence=['latency', 'rssi'], evidence_card=[2, 2])
print(cpt_reliability)



# Operations Node
cpt_operations = TabularCPD(variable='operations', variable_card=2,
                            values=[[0.9, 0.5, 0.5, 0.01],
                                    [0.1, 0.5, 0.5, 0.99]],
                            evidence=['confidentiality', 'integrity'],
                            evidence_card=[2, 2])
print(cpt_operations)


# Standards Node
cpt_standards = TabularCPD(variable='standards', variable_card=2,
                            values=[[0.9, 0.5, 0.5, 0.01],
                                    [0.1, 0.5, 0.5, 0.99]],
                            evidence=['compliance', 'threatsafe'],
                            evidence_card=[2, 2])
print(cpt_standards)

# Reuse Node
# cpt_reuse = TabularCPD(variable='reuse', variable_card=2,
#                             values=[[0.9, 0.6, 0.6, 0.01],
#                                     [0.1, 0.4, 0.4, 0.99]],
#                             evidence=['datastorage', 'datausage'],
#                             evidence_card=[2, 2])
# print(cpt_reuse)


# Protection Node
# cpt_protection = TabularCPD(variable='protection', variable_card=2,
#                             values=[[0.9, 0.6, 0.6, 0.01],
#                                     [0.1, 0.4, 0.4, 0.99]],
#                             evidence=['transparent', 'anonymize'],
#                             evidence_card=[2, 2])
# print(cpt_protection)


# ------------------------------------------------------------------------------------------------------#
# Thirdly, we start building the probability table for the three Determinant- descendant nodes with parent#
# ------------------------------------------------------------------------------------------------------#

# Robustness Node
cpt_robustness = TabularCPD(variable='robustness', variable_card=2,
                            values=[[0.6, 0.5, 0.5, 0.4],
                                    [0.4, 0.5, 0.5, 0.6]],
                            evidence=['performance', 'reliability'],
                            evidence_card=[2, 2])
print(cpt_robustness)

cpt_robustnessbest = TabularCPD(variable='robustness', variable_card=2,
                            values=[[1, 0.6, 0.6, 0.01],
                                    [0, 0.4, 0.4, 0.99]],
                            evidence=['performance', 'reliability'],
                            evidence_card=[2, 2])
print(cpt_robustnessbest)

cpt_robustnessworst = TabularCPD(variable='robustness', variable_card=2,
                            values=[[0, 0.5, 0.6, 0.5],
                                    [1, 0.5, 0.4, 0.5]],
                            evidence=['performance', 'reliability'],
                            evidence_card=[2, 2])
print(cpt_robustnessworst)

# Security Node
cpt_security = TabularCPD(variable='security', variable_card=2,
                            values=[[0.6, 0.5, 0.5, 0.4],
                                    [0.4, 0.5, 0.5, 0.6]],
                            evidence=['operations', 'standards'],
                            evidence_card=[2, 2])
print(cpt_security)
##
cpt_securitybest = TabularCPD(variable='security', variable_card=2,
                            values=[[1, 0.1, 0.1, 0.01],
                                    [0, 0.9, 0.9, 0.99]],
                            evidence=['operations', 'standards'],
                            evidence_card=[2, 2])
print(cpt_securitybest)

##
cpt_securityworst = TabularCPD(variable='security', variable_card=2,
                            values=[[0.2, 0.1, 0.1, 0.6],
                                    [0.8, 0.9, 0.9, 0.4]],
                            evidence=['operations', 'standards'],
                            evidence_card=[2, 2])
print(cpt_securityworst)

#Privacy Node
cpt_privacy = TabularCPD(variable='privacy', variable_card=2,
                         values=[[0.6, 0.3],
                                 [0.4, 0.7]],
                         evidence=['transparent'], evidence_card=[2])
print(cpt_privacy)

#Privacy Node
cpt_privacybest = TabularCPD(variable='privacy', variable_card=2,
                         values=[[0.8, 0.2],
                                 [0.2, 0.8]],
                         evidence=['transparent'], evidence_card=[2])
print(cpt_privacybest)



##
cpt_privacyworst = TabularCPD(variable='privacy', variable_card=2,
                         values=[[0.4, 0.5],
                                 [0.6, 0.5]],
                         evidence=['transparent'], evidence_card=[2])
print(cpt_privacyworst)


# cpt_privacy = TabularCPD(variable='privacy', variable_card=2,
#                             values=[[1, 0.1, 0.1, 0.01],
#                                     [0, 0.9, 0.9, 0.99]],
#                             evidence=['reuse', 'protection'],
#                             evidence_card=[2, 2])
# print(cpt_privacy)


# ------------------------------------------------------------------------------------------------------#
# Fourthly, we build the probability table for Trust#
# ------------------------------------------------------------------------------------------------------#
# Trust1 Node
cpt_trust1 = TabularCPD(variable='trust1', variable_card=2,
                       values=[[1, 0.7, 0.7, 0.01],
                                [0, 0.3, 0.3, 0.99]],
                        evidence=['robustness', 'security'],
                            #evidence=['robustness', 'security', 'privacy'],
                            evidence_card=[2,2])
print(cpt_trust1)

# Trust Node
cpt_trust = TabularCPD(variable='trust', variable_card=2,
                       values=[[0.9, 0.1, 0.1, 0.01],
                             [0.1, 0.9, 0.9, 0.99]],
                        evidence=['trust1', 'privacy'],
                            #evidence=['robustness', 'security', 'privacy'],
                            evidence_card=[2,2])
print(cpt_trust)

# cpt_trust = TabularCPD(variable='trust', variable_card=2,
#                        values=[[0.9, 0.1, 0.1, 0.01],
#                                 [0.1, 0.9, 0.9, 0.99]],
#                         evidence=['trust1', 'privacy'],
#                             #evidence=['robustness', 'security', 'privacy'],
#                             evidence_card=[2,2])
#print(cpt_trust)



# ------------------------------------------------------------------------------------------------------#
# Fifth,, we Update DAG with the CPTs
# ------------------------------------------------------------------------------------------------------#

# model = bn.make_DAG(DAG, CPD=[cpt_memory, cpt_power, cpt_rssi, cpt_latency, cpt_confidentiality,cpt_integrity, cpt_compliance, cpt_threatsafe,
#                                 cpt_reliability, cpt_performance, cpt_operations, cpt_standards,
#                               cpt_robustness,cpt_security,
#                               cpt_trust1,  cpt_trust])

modelactual = bn.make_DAG(DAG, CPD=[cpt_memory, cpt_power, cpt_rssi, cpt_latency, cpt_confidentiality,cpt_integrity, cpt_compliance,
                              cpt_threatsafe,
                              #cpt_datausage, cpt_datastorage,
                              cpt_transparent, #cpt_anonymize,
                              cpt_reliability, cpt_performance, cpt_operations, cpt_standards,
                              #cpt_reuse,
                              #cpt_protection,
                              cpt_robustness,cpt_security,cpt_privacy, cpt_trust1,
                              cpt_trust])

# Print the CPTs
bn.print_CPD(modelactual)
#

modelbest = bn.make_DAG(DAG, CPD=[cpt_memory, cpt_power, cpt_rssi, cpt_latency, cpt_confidentiality,cpt_integrity, cpt_compliance,
                              cpt_threatsafe,
                              #cpt_datausage, cpt_datastorage,
                              cpt_transparent, #cpt_anonymize,
                              cpt_reliability, cpt_performance, cpt_operations, cpt_standards,
                              #cpt_reuse,
                              #cpt_protection,
                              cpt_robustnessbest,cpt_securitybest,cpt_privacybest, cpt_trust1,
                              cpt_trust])
# Print the CPTs
bn.print_CPD(modelbest)


modelworst= bn.make_DAG(DAG, CPD=[cpt_memory, cpt_power, cpt_rssi, cpt_latency, cpt_confidentiality,cpt_integrity, cpt_compliance,
                              cpt_threatsafe,
                              #cpt_datausage, cpt_datastorage,
                              cpt_transparent, #cpt_anonymize,
                              cpt_reliability, cpt_performance, cpt_operations, cpt_standards,
                              #cpt_reuse,
                              #cpt_protection,
                              cpt_robustnessworst,cpt_securityworst,cpt_privacyworst, cpt_trust1,
                              cpt_trust])
# Print the CPTs
bn.print_CPD(modelworst)

# ------------------------------------------------------------------------------------------------------#
# Finally, we start making Inferences
# ------------------------------------------------------------------------------------------------------#

# # Make inference on robustness given power is high
q1 = bn.inference.fit(modelworst, variables=['trust'], evidence={'robustness': 0})
print(q1.df)
# q2 = bn.inference.fit(modelbest, variables=['trust1'], evidence={'robustness': 0})
# print(q2.df)
#
# q3 = bn.inference.fit(modelactual, variables=['trust'], evidence={'privacy': 1})
# print(q3.df)
q4 = bn.inference.fit(modelbest, variables=['trust'], evidence={'security': 1})
print(q4.df)


# ------------------------------------------------------------------------------------------------------#
# Lastly, we compare the models
# ------------------------------------------------------------------------------------------------------#
# Compare networks and make plot
#bn.compare_networks(modelactual, modelbest, pos=DAG['pos'])
# bn.compare_networks(modelbest, modelactual)
# bn.compare_networks(modelactual,modelbest)
bn.compare_networks(modelworst,modelbest)
# bn.compare_networks(modelworst,modelactual)
