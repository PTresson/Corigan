rm(list=ls())
setwd(getwd())
getwd()
library(ggplot2)
library(tidyr)
library(dplyr)
library(scales)
library(GGally)
library(network)
library(sna)
library(tibble)
library(tidygraph)
library(ggraph)
library(igraph)


data = read.table('inter_r_friendly.csv', sep = ',', h=T)

data = data %>%
  filter(count != 0)



graph <- graph_from_data_frame(data)
graph

data_pred = data %>%
  filter(orient == 0)
graph_pred <- graph_from_data_frame(data_pred)
data

layout1 = create_layout(graph, layout = 'linear', circular = TRUE)

ggraph(layout1) +
  #geom_node_text(aes(label = name))+
  geom_edge_arc(aes(colour = factor(relation), width = count), alpha = 0.7)+
  scale_edge_color_manual(name = 'Relation', values = c('grey50','chartreuse3','darkred','coral4','grey90'))+
  scale_edge_width(name = 'Interaction count')+
  theme(axis.line = element_blank())+
  theme(axis.text = element_blank())+
  theme(axis.ticks = element_blank())+
  theme(axis.title = element_blank())+
  theme(panel.background = element_rect(fill = 'white'))


