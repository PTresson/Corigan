rm(list=ls())
setwd("/home/tresson/resultats_02_19/interaction_matrix/plot_with_R")
getwd()
library(ggplot2)
library(tidyr)
library(dplyr)
library(scales)

data = read.table('interaction_matrix_intra.csv', sep = ',', h=T)
data

#
data = data %>%
  gather(classB, count, -X)

head(data)

data = rename(data, classA = X)

head(data)

distinct_df = data %>% distinct(classA)
distinct_df

distinct_vector = distinct_df$classA
distinct_vector

breaks = distinct_vector
labels = c('1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20')



data <- data %>% mutate(classB = factor(classB), classB = factor(classB, levels = rev(levels(classB))))

trans_log_x1 = trans_new(name = 'logXplus1', function(x) log10(x+1), function(x) (10^x)-1)


ggplot(data, aes(x=classA, y=classB, fill=count)) + 
  geom_tile(colour = 'grey50', size = 5)+
  scale_fill_continuous(trans = trans_log_x1, 
                        low = 'white', high = 'red4',
                        breaks = c(1,10,100,1000), labels = c('1','10','100','1000'),
                        name = 'Interactions')+
  scale_x_discrete(name = " ", breaks, labels)+
  scale_y_discrete(name = " ", breaks, labels)+
  theme(legend.justification = 'top')+
  theme(axis.text.x = element_text(angle = 30, hjust = 1))+
  theme(axis.text = element_text(size = 100))+
  theme(axis.title = element_text(size = 100))+
  theme(legend.title = element_text(size = 100))+
  theme(legend.text = element_text(size = 100))+
  theme(axis.ticks.length = unit(20,"point") , axis.ticks = element_line(size = 4, color = 'grey'))+
  guides(fill = guide_colourbar(barwidth = 10, barheight = 50, ticks.linewidth = 5))




