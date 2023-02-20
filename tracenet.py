# TraceNet v1.0 by Dongsh 

import numpy as np
import os
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, callbacks

import scipy.integrate as si

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_file_list(basis_dir="./", begin="", end=""):
	path_list = os.listdir(basis_dir)
	list_final = []
	for partial in path_list:
		if begin and end:
			if partial[:len(begin)] == begin and partial[-len(end):] == end:
				list_final.append(partial)
				
		elif end:
			if partial[-len(end):] == end:
				list_final.append(partial)
		
		elif begin:
			if partial[:len(begin)] == begin:
				list_final.append(partial)
				
		else:
			list_final.append(partial)
			
	return list_final
	
	
def load_model(modelFilePath):
	return keras.models.load_model(modelFilePath)
	
def extrace_baseline(model, testData, shift=10):
	
	xSpace = np.arange(0, 4096)
	rawLen = len(testData)
	testData = np.concatenate([np.zeros(shift), testData, testData[-shift:]+(testData[-1]-testData[-shift])])
	
	testData = np.interp(xSpace, np.linspace(0,4096,len(testData)),testData)
	testDataGo = testData * 1.5 / np.max(np.abs(testData))
	testDataGo = tf.convert_to_tensor(testDataGo)
	testDataGo = tf.expand_dims(testDataGo, -1)
	testDataGo = tf.expand_dims(testDataGo, 0)
	result = np.squeeze(model.predict(testDataGo)) * np.max(np.abs(testData)) / 1.5
	
#	trResult = tr.copy()
	result = np.interp(np.arange(rawLen+2*shift),np.linspace(0,rawLen+2*shift,4096), result)
	result = result[shift:-shift]
	return result
	

	
#def plot_nez(accN, accE, accZ, veloN, veloE, veloZ, corrN, corrE, corrZ, dispN, dispE, dispZ, mN, mE, mZ ,dt, fileName, manualN=[], manualE=[], manualZ=[]):
def plot_nez(accNEZ, veloNEZ, baselineNEZ, dispNEZ, offsetNEZ, dt, fileName, manualN=[], manualE=[], manualZ=[]):
#   plt.figure(figsize=(20,10))

	accN = accNEZ[0]
	accE = accNEZ[1]
	accZ = accNEZ[2]
	
	veloN = veloNEZ[0]
	veloE = veloNEZ[1]
	veloZ = veloNEZ[2]
	
	corrN = baselineNEZ[0]
	corrE = baselineNEZ[1]
	corrZ = baselineNEZ[2]
	
	dispN = dispNEZ[0]
	dispE = dispNEZ[1]
	dispZ = dispNEZ[2]
	
	mN = offsetNEZ[0]
	mE = offsetNEZ[1]
	mZ = offsetNEZ[2]
	
	
	if len(manualN) > 0:
		manualN = -si.cumtrapz(manualN, dx=dt)
		manualE = -si.cumtrapz(manualE, dx=dt)
		manualZ = si.cumtrapz(manualZ, dx=dt)

	
	plt.figure(figsize=(15,8))
	
	plt.subplot(321)
	plt.plot(np.arange(len(veloN))*dt, veloN, label='Velocity NS', lw=1, c='orange')
	plt.plot(np.arange(len(veloN))*dt, veloN - corrN, label='Corrected Velocity NS', lw=1, c='gray')
	plt.plot([0, len(veloN)*dt], [0, 0], lw=1, linestyle="--", c='k')
	plt.plot(np.arange(len(corrN))*dt, corrN, label='Baseline NS', lw=1, c='navy')
	
	plt.legend(loc='upper right')
	plt.gca().set_xlim([0, len(accE)*dt])
	__mst(fontsize=10,ylabel='Velocity (cm/s)')
	
	plt.subplot(323)
	plt.plot(np.arange(len(veloE))*dt, veloE, label='Velocity EW', lw=1, c='orange')
	plt.plot(np.arange(len(veloE))*dt, veloE - corrE, label='Corrected Velocity EW', lw=1, c='gray')
	plt.plot([0, len(veloE)*dt], [0, 0], lw=1, linestyle="--", c='k')
	plt.plot(np.arange(len(corrE))*dt, corrE, label='Baseline EW', lw=1, c='navy')
	plt.legend(loc='upper right')
	plt.gca().set_xlim([0, len(accE)*dt])
	__mst(fontsize=10,ylabel='Velocity (cm/s)')
	
	plt.subplot(325)
	plt.plot(np.arange(len(veloZ))*dt, veloZ, label='Velocity UD', lw=1, c='orange')
	plt.plot(np.arange(len(veloZ))*dt, veloZ - corrZ, label='Corrected Velocity UD', lw=1, c='gray')
	plt.plot([0, len(veloZ)*dt], [0, 0], lw=1, linestyle="--", c='k')
	plt.plot(np.arange(len(corrZ))*dt, corrZ, label='Baseline UD', lw=1, c='navy')
	plt.legend(loc='upper right')
	plt.gca().set_xlim([0, len(accE)*dt])
	__mst(fontsize=10,ylabel='Velocity (cm/s)', xlabel='Time (s)')
	

	dispRawN = si.cumtrapz(veloN, dx=dt)
	dispRawE = si.cumtrapz(veloE, dx=dt)
	dispRawZ = si.cumtrapz(veloZ, dx=dt)
	
	boostCoef = 1.5
	
	plt.subplot(3,2,2)
	
	if len(manualN) > 0:
		plt.plot(np.arange(len(manualN))*dt, manualN, label='Manual corr. NS', lw=1, c='gray')
		
	plt.plot(np.arange(len(dispRawN))*dt, dispRawN, label='Non-corrected NS', lw=1, c='orange')
	plt.plot(np.arange(len(dispN))*dt, dispN, label='Corrected NS', lw=1, c='navy')

	plt.plot([0, len(dispN)*dt], [0, 0], lw=1, linestyle="--", c='k', alpha=.7)
	plt.plot([0, len(dispN)*dt], [mN, mN], lw=1, linestyle="--", c='red')
	plt.legend(loc='upper right')
	plt.gca().set_xlim([0, len(accE)*dt])

	
	__mst(fontsize=10, ylabel='Displacement (cm)')

	plt.gca().set_ylim(boostCoef*np.min(np.hstack([dispN, manualN])), boostCoef*np.max(np.hstack([dispN,manualN])))
	
	plt.subplot(3,2,4)
	if len(manualN) > 0:
		plt.plot(np.arange(len(manualE))*dt, manualE, label='Manual corr. EW', lw=1, c='gray')
		
	plt.plot(np.arange(len(dispRawE))*dt, dispRawE, label='Non-corrected EW', lw=1, c='orange')
	plt.plot(np.arange(len(dispE))*dt, dispE, label='Corrected EW', lw=1, c='navy')
	plt.plot([0, len(dispE)*dt], [0, 0], lw=1, linestyle="--", c='k', alpha=.7)
	plt.plot([0, len(dispE)*dt], [mE, mE], lw=1, linestyle="--", c='red')
	plt.legend(loc='upper right')
	plt.gca().set_xlim([0, len(accE)*dt])
	__mst(fontsize=10, ylabel='Displacement (cm)')
	plt.gca().set_ylim(boostCoef*np.min(np.hstack([dispE, manualE])), boostCoef*np.max(np.hstack([dispE,manualE])))
	
	plt.subplot(326)
	if len(manualZ) > 0:
		plt.plot(np.arange(len(manualZ))*dt, manualZ, label='Manual corr. UD', lw=1, c='gray')
	plt.plot(np.arange(len(dispRawZ))*dt, dispRawZ, label='Non-corrected UD', lw=1, c='orange')
	plt.plot(np.arange(len(dispZ))*dt, dispZ, label='Corrected UD', lw=1, c='navy')
	plt.plot([0, len(dispZ)*dt], [0, 0], lw=1, linestyle="--", c='k', alpha=.7)
	plt.plot([0, len(dispZ)*dt], [mZ, mZ], lw=1, linestyle="--", c='red')
	
	plt.legend(loc='upper right')
	plt.gca().set_xlim([0, len(accE)*dt])
#   plt.gca().set_ylim([boostCoef*np.min(dispZ), boostCoef*np.max(dispZ)])
#  plt.gca().set_ylim([mZ/2 - boostCoef*mZ, mZ/2 + boostCoef*mZ])
	plt.gca().set_ylim(boostCoef*np.min(np.hstack([dispZ, manualZ])), boostCoef*np.max(np.hstack([dispZ,manualZ])))
	__mst(fontsize=10,xlabel='Time (s)', ylabel='Displacement (cm)')
	
	plt.savefig(fileName, dpi=600)
	plt.close()


# multi_set_tool
def __mst(ax=[], label=[], xlabel="", ylabel="",title=[], xlim=[], ylim=[], style=[], fontsize=8, width_line=1 ,legend=False, rotate=False, axis_width=1, half_axis=False):
	
	if not ax:
		ax = plt.gca()
		
	if style == "Times":
		font1 = {'family' : 'stix', 'size': fontsize,}
		font2 = {'family' : 'stix', 'size': fontsize*1.2,}
		font3 = {'family' : 'stix', 'size': fontsize*1.4,}
		plt.tick_params(labelsize=fontsize)
		labels = ax.get_xticklabels() + ax.get_yticklabels()
		[label.set_fontname('stix') for label in labels]
		
	elif style == "Arial":
		font1 = {'family' : 'Arial', 'size': fontsize,}
		font2 = {'family' : 'Arial', 'size': fontsize*1.2,}
		font3 = {'family' : 'Arial', 'size': fontsize*1.4,}
		plt.tick_params(labelsize=fontsize)
		labels = ax.get_xticklabels() + ax.get_yticklabels()
		[label.set_fontname('Arial') for label in labels]
		
	else:
		font1 = {'size': fontsize,}
		font2 = {'size': fontsize*1.2,}
		font3 = {'size': fontsize*1.4,}
		plt.tick_params(labelsize=fontsize)
		
	if half_axis:
		for axis in ['bottom','left']:
			ax.spines[axis].set_linewidth(axis_width)
		for axis in ['top', 'right']:
			ax.spines[axis].set_linewidth(0)
	else:
		for axis in ['top','bottom','left','right']:
			ax.spines[axis].set_linewidth(axis_width)
	
	if title:
		ax.set_title(title,font2)

	if len(label) == 2:
		ax.set_xlabel(label[0],font2)
		ax.set_ylabel(label[1],font2)
	
	if xlabel:
		ax.set_xlabel(xlabel, font2)
	
	if ylabel:
		ax.set_ylabel(ylabel, font2)
	
	if legend:
		legend = ax.legend(prop=font1)
			
	if len(xlim) == 2:
		ax.set_xlim(xlim)

	if rotate:
		ax.set_xticklabels(ax.get_xticks(),rotation=45)
	
	if len(ylim) == 2:
		ax.set_ylim(ylim)
