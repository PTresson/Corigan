rm(list=ls())
setwd("/home/tresson/resultats_02_19/abundance_over_time")
getwd()
library(ggplot2)

data = read.table('object_count_tot.csv', sep = ',', h=T)
data$date = strptime(data$date, "%Y:%m:%d %H:%M:%S")
data$date = as.POSIXct(data$date)

head(data)

start_date = "2000-01-01 00:00:00 CET"
end_date = "2018-11-13 20:30:00 CET"

data = data[data$date < end_date,]

Metamasius_label = expression(paste(italic("Metamasius")," larva"))
Nylanderia_label = expression(paste(italic("Nylanderia")," msp1"))
Odontomachus_label = expression(paste(italic("Odontomachus bauri"),""))
pheidole_major_label = expression(paste(italic("Pheidole radoszkowskii")," major"))
pheidole_minor_label = expression(paste(italic("Pheidole radoszkowskii")," minor"))
Solenopsis_minor_label = expression(paste(italic("Solenopsis geminata")," minor"))

plot_abundance = ggplot(data = data, aes(x = date, y = count, linetype = detec))+
  geom_line(data = data[data$detec == 'detection',], aes(color = obj_class), linetype =1, size = 7)+
  geom_line(data = data[data$detec == 'GT',], aes(x = date, y = count,color = obj_class), linetype = 2, size = 7)+
  theme(panel.background = element_rect(fill = "white"))+
  theme(panel.grid.major = element_line(colour = "lightgrey"))+
  #theme(panel.grid.minor = element_line(colour = "lightgrey"))+
  theme(axis.line = element_line(color = "black"))+
  scale_color_manual(values = c('chartreuse4', 'dodgerblue3','firebrick3','purple4','purple3','goldenrod3'), 
                     labels = c(Metamasius_label, Nylanderia_label, Odontomachus_label, pheidole_major_label, pheidole_minor_label, Solenopsis_minor_label), 
                     name = "Object class")+
  theme(panel.background = element_rect(fill = "white"))+
  theme(panel.grid.major = element_line(colour = "lightgrey", size = 7))+
  #theme(panel.grid.minor = element_line(colour = "lightgrey"))+
  theme(axis.line = element_line(color = "black", size = 10))+
  theme(axis.text = element_text(size = 100))+
  theme(axis.title = element_text(size = 100))+
  theme(legend.title = element_text(size = 100))+
  theme(legend.text = element_text(size = 100))+
  theme(axis.ticks.length = unit(20,"point") , axis.ticks = element_line(size = 1, color = 'grey'))+
  guides(colour = guide_legend(override.aes =  list(size=50)))+
  xlab("Time (HH:mm)")+
  ylab("Abundance (individuals)")#+
  #guides(color=guide_legend(title = "Object class"))
 
plot_abundance


plot_abundance = ggplot(data = data, aes(x = date, y = count, linetype = detec))+
  geom_line(data = data, aes(color = obj_class, linetype = detec), size = 7)+
  theme(panel.background = element_rect(fill = "white"))+
  theme(panel.grid.major = element_line(colour = "lightgrey"))+
  #theme(panel.grid.minor = element_line(colour = "lightgrey"))+
  theme(axis.line = element_line(color = "black"))+
  scale_color_manual(values = c('chartreuse4', 'dodgerblue3','firebrick3','purple4','purple3','goldenrod3'), 
                     labels = c(Metamasius_label, Nylanderia_label, Odontomachus_label, pheidole_major_label, pheidole_minor_label, Solenopsis_minor_label), 
                     name = "\nObject class")+
  scale_linetype_manual(values = c(1, 2), 
                     labels = c("Detection", "Ground truth"), 
                     name = "   ")+
  theme(panel.background = element_rect(fill = "white"))+
  theme(panel.grid.major = element_line(colour = "lightgrey", size = 7))+
  #theme(panel.grid.minor = element_line(colour = "lightgrey"))+
  theme(axis.line = element_line(color = "black", size = 10))+
  theme(axis.text = element_text(size = 100))+
  theme(axis.title = element_text(size = 100))+
  theme(legend.title = element_text(size = 100))+
  theme(legend.text = element_text(size = 100))+
  theme(legend.key.width = unit(250,'point'))+
  theme(legend.text.align = 0)+
  theme(axis.ticks.length = unit(20,"point") , axis.ticks = element_line(size = 1, color = 'grey'))+
  guides(colour = guide_legend(override.aes =  list(size=7)),linetype = guide_legend(override.aes =  list(size=7)))+
  xlab("Time (hh:mm)")+
  ylab("Abundance (individuals)")#+
#guides(color=guide_legend(title = "Object class"))

plot_abundance

####
length(data$date)
data2 = data[data$obj_class != 'Solenopsis_geminata_minor',]
unique(data2$obj_class)
unique(data$obj_class)

plot_abundance = ggplot(data = data2, aes(x = date, y = count, linetype = detec))+
  geom_line(data = data2, aes(color = obj_class, linetype = detec), size = 7)+
  theme(panel.background = element_rect(fill = "white"))+
  theme(panel.grid.major = element_line(colour = "lightgrey"))+
  #theme(panel.grid.minor = element_line(colour = "lightgrey"))+
  theme(axis.line = element_line(color = "black"))+
  scale_color_manual(values = c('chartreuse4', 'dodgerblue3','firebrick3','purple4','purple3'), 
                     labels = c(Metamasius_label, Nylanderia_label, Odontomachus_label, pheidole_major_label, pheidole_minor_label), 
                     name = "\nObject class")+
  scale_linetype_manual(values = c(1, 2), 
                        labels = c("Detection", "Ground truth"), 
                        name = "   ")+
  theme(panel.background = element_rect(fill = "white"))+
  theme(panel.grid.major = element_line(colour = "lightgrey", size = 7))+
  #theme(panel.grid.minor = element_line(colour = "lightgrey"))+
  theme(axis.line = element_line(color = "black", size = 10))+
  theme(axis.text = element_text(size = 100))+
  theme(axis.title = element_text(size = 100))+
  theme(legend.title = element_text(size = 100))+
  theme(legend.text = element_text(size = 100))+
  theme(legend.key.width = unit(250,'point'))+
  theme(legend.text.align = 0)+
  theme(axis.ticks.length = unit(20,"point") , axis.ticks = element_line(size = 1, color = 'grey'))+
  guides(colour = guide_legend(override.aes =  list(size=7)),linetype = guide_legend(override.aes =  list(size=7)))+
  xlab("Time (hh:mm)")+
  ylab("Abundance (individuals)")#+
#guides(color=guide_legend(title = "Object class"))

plot_abundance

####################################################################### all night

data = read.table('object_count_tot.csv', sep = ',', h=T)
data$date = strptime(data$date, "%Y:%m:%d %H:%M:%S")
data$date = as.POSIXct(data$date)

head(data)

start_date = "2000-01-01 00:00:00 CET"
end_date = "2018-11-14 06:30:00 CET"

data = data[data$date < end_date,]



y_title = bquote(atop(italic("Pheidole radoszkowskii")," abundance (individuals)"))
pheidole_major_label = expression(paste(italic("Pheidole radoszkowskii")," major"))
pheidole_minor_label = expression(paste(italic("Pheidole radoszkowskii")," minor"))

plot_abundance_night = ggplot(data = data, aes(x = date, y = count))+
  geom_line(data = data[data$detec == 'detection'& (data$obj_class == 'Pheidole_radoszkowskii_minor' | data$obj_class == 'Pheidole_radoszkowskii_major'),], aes(color = obj_class), size = 7)+
  #geom_line(data = data[data$detec == 'detection'& data$obj_class == 'Pheidole_radoszkowskii_major',], aes(color = obj_class), color = 'purple4')+
  scale_color_manual(values = c('firebrick', 'purple3'), labels = c(pheidole_major_label,pheidole_minor_label), name = "Object class")+
  theme(panel.background = element_rect(fill = "white"))+
  theme(panel.grid.major = element_line(colour = "lightgrey", size = 7))+
  theme(panel.grid.minor = element_line(colour = "lightgrey", size = 5))+
  theme(axis.line = element_line(color = "black", size = 10))+
  theme(legend.position = c(.65,.80))+
  theme(axis.text = element_text(size = 100))+
  theme(axis.title = element_text(size = 100))+
  theme(legend.title = element_text(size = 100))+
  theme(legend.text = element_text(size = 100))+
  theme(axis.ticks.length = unit(20,"point") , axis.ticks = element_line(size = 1, color = 'grey'))+
  theme(legend.key.width = unit(250,'point'))+
  guides(colour = guide_legend(override.aes =  list(size=7)))+
  xlab("Time (hh:mm)")+
  ylab(y_title)

plot_abundance_night

pheidole_major_label = "major"
pheidole_minor_label = "minor"

plot_abundance_night = ggplot(data = data, aes(x = date, y = count))+
  geom_line(data = data[data$detec == 'detection'& (data$obj_class == 'Pheidole_radoszkowskii_minor' | data$obj_class == 'Pheidole_radoszkowskii_major'),], aes(color = obj_class), size = 7)+
  #geom_line(data = data[data$detec == 'detection'& data$obj_class == 'Pheidole_radoszkowskii_major',], aes(color = obj_class), color = 'purple4')+
  scale_color_manual(values = c('firebrick', 'purple3'), labels = c(pheidole_major_label,pheidole_minor_label), name = "Ant class")+
  theme(panel.background = element_rect(fill = "white"))+
  theme(panel.grid.major = element_line(colour = "lightgrey", size = 7))+
  theme(panel.grid.minor = element_line(colour = "lightgrey", size = 3))+
  theme(axis.line = element_line(color = "black", size = 10))+
  theme(legend.position = c(.684,.796))+
  theme(axis.text = element_text(size = 100))+
  theme(axis.title = element_text(size = 100))+
  theme(legend.title = element_text(size = 100))+
  theme(legend.text = element_text(size = 100))+
  theme(axis.ticks.length = unit(20,"point") , axis.ticks = element_line(size = 1, color = 'grey'))+
  theme(legend.key.width = unit(300,'point'))+
  theme(legend.key.size = unit(150,'point'))+
  theme(legend.key = element_rect(fill = "white"))+
  guides(colour = guide_legend(override.aes =  list(size=7)))+
  xlab("Time (hh:mm)")+
  ylab(y_title)

plot_abundance_night



###########""

plot_abundance_night = ggplot(data = data, aes(x = date, y = count))+
  geom_line(data = data[data$detec == 'detection'& data$obj_class == 'Metamasius_larva',], aes(color = obj_class), color = 'chartreuse4')+
  geom_line(data = data[data$detec == 'GT'& data$obj_class == 'Metamasius_larva',], aes(color = obj_class), color = 'chartreuse4', linetype = 2)+
  theme(panel.background = element_rect(fill = "white"))+
  theme(panel.grid.major = element_line(colour = "lightgrey"))+
  #theme(panel.grid.minor = element_line(colour = "lightgrey"))+
  theme(axis.line = element_line(color = "black"))+
  xlab("Time (HH:mm)")+
  ylab("Metamasius larva abundance (individuals)")

plot_abundance_night
