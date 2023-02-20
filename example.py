

import tracenet
import obspy
import numpy as np
import os 
import time

### Path and File Settings
# path settings
acc_sac_data_file_path = './data/'    # raw acceleration data file path 
figure_save_path = './result_figure'   # path for output figures 
offset_result_save_path = './result_offset.txt'   # path for offset values
correct_velo_save_path = './corr_velos'  # path for output corrected velocities 

# Set TraceNet model path
model_file='./TraceNet_Finished.h5'
TraceNet = tracenet.load_model(model_file)

### Start TraceNet correction

# search files on UD channel as the file list for correction
fileList = tracenet.get_file_list(acc_sac_data_file_path, end='UD.SAC')

if not os.path.exists(figure_save_path):
   os.mkdir(figure_save_path)
   
if not os.path.exists(correct_velo_save_path):
   os.mkdir(correct_velo_save_path)
   
if len(offset_result_save_path)>0:
   file = open(offset_result_save_path,'w')

t1 = time.time()
for fileName in fileList:
   
   # Read raw acceleration data in N-E-Z channel
   UDfullName = os.path.join(acc_sac_data_file_path,fileName)
   EWfullName = UDfullName[:-6]+'EW.SAC'
   NSfullName = UDfullName[:-6]+'NS.SAC'
   
   udTr = obspy.read(UDfullName)[0]
   ewTr = obspy.read(EWfullName)[0]
   nsTr = obspy.read(NSfullName)[0]
   st = obspy.Stream([nsTr,ewTr,udTr])

   baselineNEZ = []
   disp_corrected = []
   offsetNEZ = []
   accNEZ = []
   
   if len(offset_result_save_path)>0:
      file.write(st[0].stats['station']+'\t')
   
   print('TraceNet work at station:', st[0].stats['station'])
   for tr in st:
      print('-- Channel:', tr.stats['channel'])
     
      # inital correction on raw acceleration data
      avgLen = int(10 / float(st[0].stats['delta']))
      tr.data -= np.mean(tr.data[:avgLen])
      accNEZ.append(tr.data)
      
      # 1st integration (to velocity)
      tr.integrate(method='cumtrapz')  
      velo_corrected = tr.copy()

      # TraceNet correction in single trace
      baseline = tracenet.extrace_baseline(TraceNet, tr.data)
      baselineNEZ.append(baseline)
      velo_corrected.data -= baseline
      
      # 2nd integration (to displacement)
      velo_corrected.integrate(method='cumtrapz')   
      disp_corrected.append(velo_corrected.copy())

      # calculate final ground offset 
      offset_value = np.mean(velo_corrected.data[-int(5 / float(st[0].stats['delta'])):])
      offsetNEZ.append(offset_value)
      
      # save result velocity and offset
      if len(correct_velo_save_path) > 0:
         tr.write(os.path.join(correct_velo_save_path, 'velo_corr_'+tr.stats['station']+'-'+tr.stats['channel'] + '.sac'))
      
      if len(offset_result_save_path)>0:
         file.write(str(offset_value)+'\t')
      
   # plot figure
   outFileName = os.path.join(figure_save_path, tr.stats['station']+'_TraceNet.pdf')
   veloNEZ = [st[0].data, st[1].data, st[2].data]
   dispNEZ = [disp_corrected[0].data, disp_corrected[1].data, disp_corrected[2].data]
   tracenet.plot_nez(accNEZ, veloNEZ , baselineNEZ, dispNEZ, offsetNEZ, float(st[0].stats['delta']), outFileName)

   if len(offset_result_save_path)>0:
      file.write('\n')
   

if len(offset_result_save_path)>0:
   file.close()

print('\n [total time used:', time.time()-t1, 'sec]')


   