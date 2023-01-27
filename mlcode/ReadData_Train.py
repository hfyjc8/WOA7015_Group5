import wfdb
import matplotlib.pyplot as plt
import pandas as pd
from wfdb import processing

"""
infant     婴儿
frequency      频率
respiration     呼吸
corrected peak inds   校正峰值indds
"""
data_dir = "/Users/apple/Desktop/content/physionet.org/files/picsdb/1.0.0"
ecg_record = wfdb.rdsamp(f"{data_dir}/infant1_ecg")  # infant1、2、_ecg
plt.plot(ecg_record[0][:500])  # ECG
# plt.show()

data_dir = "/Users/apple/Desktop/content/physionet.org/files/picsdb/1.0.0"
respiration1 = wfdb.rdsamp(f"{data_dir}/infant1_resp")  # 2 、3、4
respiration1  # 呼吸 心电图
# print('respiration1', respiration1)
# print('============ ============')

resampledsignal = processing.resample_sig(respiration1[0][:, 0], 500, 50)

resampledsignal  # 重新取样信号
# print("resampledsignal", resampledsignal)  # resampled signal是一个个 array 组成的
ecg_record[0]
# print('ecg_record[0]', ecg_record[0])

import wfdb.processing as processing


def peaks_hr(sig, peak_inds, fs, title, figsize=(20, 10), saveto=None):
    "Plot a signal with its peaks and heart rate"
    # Calculate heart rate
    hrs = processing.hr.compute_hr(sig_len=sig.shape[0], qrs_inds=peak_inds, fs=fs)
    return hrs


# Load the WFDB record and the physical samples
record = wfdb.rdrecord(f"{data_dir}/infant1_ecg", channels=[0])  ###   改的ecg  500->250
# Use the GQRS algorithm to detect QRS locations in the first channel
qrs_inds = processing.qrs.gqrs_detect(sig=record.p_signal[:, 0], fs=record.fs)

# Plot results
peaks_hr(sig=record.p_signal, peak_inds=qrs_inds, fs=record.fs,
         title="GQRS peak detection on infant1ECG")
# Correct the peaks shifting them to local maxima
min_bpm = 20
max_bpm = 230
# min_gap = record.fs * 60 / min_bpm
# Use the maximum possible bpm as the search radius
search_radius = int(record.fs * 60 / max_bpm)

# corrected_peak_inds
corrected_peak_inds = processing.peaks.correct_peaks(record.p_signal[:, 0],
                                                     peak_inds=qrs_inds,
                                                     search_radius=search_radius,
                                                     smooth_window_size=150)

# print('corrected_peak_inds=', corrected_peak_inds)
# Display results
# print('Corrected GQRS detected peak indices:', sorted(corrected_peak_inds)) #矫正峰值
heart_rate = peaks_hr(sig=record.p_signal, peak_inds=sorted(corrected_peak_inds), fs=record.fs,
                      title="Corrected GQRS peak detection on infant1ECG")


# print('heart_rate', heart_rate)  # heart_rate 是一个array

# resampledsignal_2d
def calculatehrin10sec(record, corrected_peak_inds):
    i = 0
    j = 0
    beatarray = []
    while i < len(record.p_signal):
        numofbeatin10sec = 0

        while corrected_peak_inds[j] < (i + ((10 / (1 / record.fs)) - 1)):
            numofbeatin10sec = numofbeatin10sec + 1
            j = j + 1
            if j >= len(corrected_peak_inds):
                break
        beatarray = beatarray + [numofbeatin10sec * 6]
        i = i + (10 / (1 / record.fs))
    return beatarray


hrarray = calculatehrin10sec(record, corrected_peak_inds)  # 我需要把 infan 1-6 变成一个datafram 7-10是另一个datafram contat 合成的意思
# print('hrarray=',hrarray)                                 #after contat 完之后 是normalization  如何进行： 23号，
resample_array = resampledsignal[0][:len(hrarray) * 500]
resampledsignal_2d = resample_array.reshape(-1, 500)
# print('resampledsignal_2d', resampledsignal_2d)
# new_heart_rate
new_heart_rate = heart_rate[::50]
# print('new_heart_rate', new_heart_rate)
df1 = pd.DataFrame(resampledsignal_2d)
# insert harray column
df1 = df1.add_prefix('respiration')
df1.insert(500, 'hrarray', hrarray)
# 显示 DataFrame
# print(df1)
print(len(df1))
print('df1结束-----------------')

'''
df2
'''
data_dir = "/Users/apple/Desktop/content/physionet.org/files/picsdb/1.0.0"
ecg_record = wfdb.rdsamp(f"{data_dir}/infant2_ecg")  # infant1、2、_ecg
plt.plot(ecg_record[0][:500])  # ECG
# plt.show()

data_dir = "/Users/apple/Desktop/content/physionet.org/files/picsdb/1.0.0"
respiration1 = wfdb.rdsamp(f"{data_dir}/infant2_resp")  # 2 、3、4
respiration1  # 呼吸 心电图

# resampledsignal = processing.resample_sig(respiration1[0][:, 0], 500, 50)
# resampledsignal  # 重新取样信号

# print("resampledsignal", resampledsignal)  # resampled signal是一个个 array 组成的
ecg_record[0]
# print('ecg_record[0]', ecg_record[0])

import wfdb.processing as processing

# Load the WFDB record and the physical samples
record = wfdb.rdrecord(f"{data_dir}/infant2_ecg", channels=[0])  ###   改的ecg  500->250
# Use the GQRS algorithm to detect QRS locations in the first channel
qrs_inds = processing.qrs.gqrs_detect(sig=record.p_signal[:, 0], fs=record.fs)

# # Plot results
# Load the WFDB record and the physical samples
record = wfdb.rdrecord(f"{data_dir}/infant2_ecg", channels=[0])


def findqrs(record):
    # Use the GQRS algorithm to detect QRS locations in the first channel
    qrs_inds = processing.qrs.gqrs_detect(sig=record.p_signal[:, 0], fs=record.fs)

    # Correct the peaks shifting them to local maxima
    min_bpm = 20
    max_bpm = 230
    # min_gap = record.fs * 60 / min_bpm
    # Use the maximum possible bpm as the search radius
    search_radius = int(record.fs * 60 / max_bpm)
    corrected_peak_inds = processing.peaks.correct_peaks(record.p_signal[:, 0],
                                                         peak_inds=qrs_inds,
                                                         search_radius=search_radius,
                                                         smooth_window_size=150)
    return corrected_peak_inds


corrected_peak_inds = findqrs(record)


# resampledsignal_2d
def calculatehrin10sec(record, corrected_peak_inds):
    i = 0
    j = 0
    beatarray = []
    while i < len(record.p_signal):
        numofbeatin10sec = 0

        while corrected_peak_inds[j] < (i + ((10 / (1 / record.fs)) - 1)):
            numofbeatin10sec = numofbeatin10sec + 1
            j = j + 1
            if j >= len(corrected_peak_inds):
                break
        beatarray = beatarray + [numofbeatin10sec * 6]
        i = i + (10 / (1 / record.fs))
    return beatarray


# print('respiration1=\n', respiration1)

hrarray = calculatehrin10sec(record, corrected_peak_inds)  # 我需要把 infan 1-6 变成一个datafram 7-10是另一个datafram contat 合成的意思
# print('hrarray=\n', hrarray)  # after contat 完之后 是normalization  如何进行： 23号，
# print('len(hrarray)=\n', len(hrarray))

print('len(respiration[0])=', len(respiration1[0]))  # respiration1[0] 表示二维数组的第一个元素
'''
取出 respiration1[0] 数组的前 len(hrarray) * 500 个元素，并赋值给 resample_array 变量/
取出了hrarray长度对应的呼吸信号数据，并赋值给resample_array.
'''
if (len(hrarray) * 500) > len(respiration1[0]):
    a = len(respiration1[0]) % 500
    resampleArray = respiration1[0][0:(len(respiration1[0]) - a)]
    hrarray = hrarray[0:int(len(resampleArray) / 500)]

else:
    resampleArray = respiration1[0][0:(len(hrarray) * 500)]

resampledsignal_2d = resampleArray.reshape(-1, 500)

df2 = pd.DataFrame(resampledsignal_2d)
# insert harray column
df2 = df2.add_prefix('respiration')
df2.insert(500, 'hrarray', hrarray)
# 显示 DataFrame
# print(df2)
print(len(df2))
print('df2 finished-------------------')

'''
df3
'''

data_dir = "/Users/apple/Desktop/content/physionet.org/files/picsdb/1.0.0"
ecg_record = wfdb.rdsamp(f"{data_dir}/infant3_ecg")  # infant1、2、_ecg
plt.plot(ecg_record[0][:500])  # ECG
# plt.show()

data_dir = "/Users/apple/Desktop/content/physionet.org/files/picsdb/1.0.0"
respiration1 = wfdb.rdsamp(f"{data_dir}/infant3_resp")  # 2 、3、4
respiration1  # 呼吸 心电图

# resampledsignal = processing.resample_sig(respiration1[0][:, 0], 500, 50)
# resampledsignal  # 重新取样信号

ecg_record[0]

import wfdb.processing as processing

# Load the WFDB record and the physical samples
record = wfdb.rdrecord(f"{data_dir}/infant3_ecg", channels=[0])  ###   改的ecg  500->250
# Use the GQRS algorithm to detect QRS locations in the first channel
qrs_inds = processing.qrs.gqrs_detect(sig=record.p_signal[:, 0], fs=record.fs)

# # Plot results
# Load the WFDB record and the physical samples
record = wfdb.rdrecord(f"{data_dir}/infant3_ecg", channels=[0])


def findqrs(record):
    # Use the GQRS algorithm to detect QRS locations in the first channel
    qrs_inds = processing.qrs.gqrs_detect(sig=record.p_signal[:, 0], fs=record.fs)

    # Correct the peaks shifting them to local maxima
    min_bpm = 20
    max_bpm = 230
    # min_gap = record.fs * 60 / min_bpm
    # Use the maximum possible bpm as the search radius
    search_radius = int(record.fs * 60 / max_bpm)
    corrected_peak_inds = processing.peaks.correct_peaks(record.p_signal[:, 0],
                                                         peak_inds=qrs_inds,
                                                         search_radius=search_radius,
                                                         smooth_window_size=150)
    return corrected_peak_inds


corrected_peak_inds = findqrs(record)


# Display results
# resampledsignal_2d
def calculatehrin10sec(record, corrected_peak_inds):
    i = 0
    j = 0
    beatarray = []
    while i < len(record.p_signal):
        numofbeatin10sec = 0

        while corrected_peak_inds[j] < (i + ((10 / (1 / record.fs)) - 1)):
            numofbeatin10sec = numofbeatin10sec + 1
            j = j + 1
            if j >= len(corrected_peak_inds):
                break
        beatarray = beatarray + [numofbeatin10sec * 6]
        i = i + (10 / (1 / record.fs))
    return beatarray


hrarray = calculatehrin10sec(record, corrected_peak_inds)  # 我需要把 infan 1-6 变成一个datafram 7-10是另一个datafram contat 合成的意思
# print('hrarray=\n', hrarray)  # after contat 完之后 是normalization  如何进行： 23号，

# print('len(respiration[0])=', len(respiration1[0]))  # respiration1[0] 表示二维数组的第一个元素
'''
取出 respiration1[0] 数组的前 len(hrarray) * 500 个元素，并赋值给 resample_array 变量/
取出了hrarray长度对应的呼吸信号数据，并赋值给resample_array.
'''

if (len(hrarray) * 500) > len(respiration1[0]):
    a = len(respiration1[0]) % 500
    resampleArray = respiration1[0][0:(len(respiration1[0]) - a)]
    hrarray = hrarray[0:int(len(resampleArray) / 500)]

else:
    resampleArray = respiration1[0][0:(len(hrarray) * 500)]

resampledsignal_2d = resampleArray.reshape(-1, 500)
# print('resampledsignal_2d=\n', resampledsignal_2d)

df3 = pd.DataFrame(resampledsignal_2d)
# insert harray column
df3 = df3.add_prefix('respiration')
df3.insert(500, 'hrarray', hrarray)
# 显示 DataFrame
print(len(df3))
print('df3 finished-----------')

'''
df4
'''

data_dir = "/Users/apple/Desktop/content/physionet.org/files/picsdb/1.0.0"
ecg_record = wfdb.rdsamp(f"{data_dir}/infant4_ecg")  # infant1、2、_ecg
plt.plot(ecg_record[0][:500])  # ECG
# plt.show()

data_dir = "/Users/apple/Desktop/content/physionet.org/files/picsdb/1.0.0"
respiration1 = wfdb.rdsamp(f"{data_dir}/infant4_resp")  # 2 、3、4
respiration1  # 呼吸 心电图
  # 重新取样信号

# print("resampledsignal", resampledsignal)  # resampled signal是一个个 array 组成的
ecg_record[0]

# Load the WFDB record and the physical samples
record = wfdb.rdrecord(f"{data_dir}/infant4_ecg", channels=[0])  ###   改的ecg  500->250
# Use the GQRS algorithm to detect QRS locations in the first channel
qrs_inds = processing.qrs.gqrs_detect(sig=record.p_signal[:, 0], fs=record.fs)

# # Plot results
# Load the WFDB record and the physical samples
record = wfdb.rdrecord(f"{data_dir}/infant4_ecg", channels=[0])


def findqrs(record):
    # Use the GQRS algorithm to detect QRS locations in the first channel
    qrs_inds = processing.qrs.gqrs_detect(sig=record.p_signal[:, 0], fs=record.fs)

    # Correct the peaks shifting them to local maxima
    min_bpm = 20
    max_bpm = 230
    # min_gap = record.fs * 60 / min_bpm
    # Use the maximum possible bpm as the search radius
    search_radius = int(record.fs * 60 / max_bpm)
    corrected_peak_inds = processing.peaks.correct_peaks(record.p_signal[:, 0],
                                                         peak_inds=qrs_inds,
                                                         search_radius=search_radius,
                                                         smooth_window_size=150)
    return corrected_peak_inds


corrected_peak_inds = findqrs(record)


# resampledsignal_2d
def calculatehrin10sec(record, corrected_peak_inds):
    i = 0
    j = 0
    beatarray = []
    while i < len(record.p_signal):
        numofbeatin10sec = 0

        while corrected_peak_inds[j] < (i + ((10 / (1 / record.fs)) - 1)):
            numofbeatin10sec = numofbeatin10sec + 1
            j = j + 1
            if j >= len(corrected_peak_inds):
                break
        beatarray = beatarray + [numofbeatin10sec * 6]
        i = i + (10 / (1 / record.fs))
    return beatarray


hrarray = calculatehrin10sec(record, corrected_peak_inds)  # 我需要把 infan 1-6 变成一个datafram 7-10是另一个datafram contat 合成的意思
# print('hrarray=\n', hrarray)  # after contat 完之后 是normalization  如何进行： 23号，

# print('len(respiration[0])=', len(respiration1[0]))  # respiration1[0] 表示二维数组的第一个元素
'''
取出 respiration1[0] 数组的前 len(hrarray) * 500 个元素，并赋值给 resample_array 变量/
取出了hrarray长度对应的呼吸信号数据，并赋值给resample_array.
'''

if (len(hrarray) * 500) > len(respiration1[0]):
    a = len(respiration1[0]) % 500
    resampleArray = respiration1[0][0:(len(respiration1[0]) - a)]
    hrarray = hrarray[0:int(len(resampleArray) / 500)]

else:
    resampleArray = respiration1[0][0:(len(hrarray) * 500)]

resampledsignal_2d = resampleArray.reshape(-1, 500)

df4 = pd.DataFrame(resampledsignal_2d)
# insert harray column
df4 = df4.add_prefix('respiration')
df4.insert(500, 'hrarray', hrarray)
# 显示 DataFrame
print(len(df4))
print('df4 finished--------------')

'''
df5
'''

data_dir = "/Users/apple/Desktop/content/physionet.org/files/picsdb/1.0.0"
ecg_record = wfdb.rdsamp(f"{data_dir}/infant5_ecg")  # infant1、2、_ecg
plt.plot(ecg_record[0][:500])  # ECG
# plt.show()

data_dir = "/Users/apple/Desktop/content/physionet.org/files/picsdb/1.0.0"
respiration1 = wfdb.rdsamp(f"{data_dir}/infant5_resp")  # 2 、3、4
respiration1  # 呼吸 心电图
# print('respiration1', respiration1)
# print('============ ============')

from wfdb import processing

resampledsignal = processing.resample_sig(respiration1[0][:, 0], 500, 50)

resampledsignal  # 重新取样信号
# print("resampledsignal", resampledsignal)  # resampled signal是一个个 array 组成的
ecg_record[0]
# print('ecg_record[0]', ecg_record[0])

import wfdb.processing as processing


def peaks_hr(sig, peak_inds, fs, title, figsize=(20, 10), saveto=None):
    "Plot a signal with its peaks and heart rate"
    # Calculate heart rate
    hrs = processing.hr.compute_hr(sig_len=sig.shape[0], qrs_inds=peak_inds, fs=fs)
    return hrs


# Load the WFDB record and the physical samples
record = wfdb.rdrecord(f"{data_dir}/infant5_ecg", channels=[0])  ###   改的ecg  500->250
# Use the GQRS algorithm to detect QRS locations in the first channel
qrs_inds = processing.qrs.gqrs_detect(sig=record.p_signal[:, 0], fs=record.fs)

# Plot results
peaks_hr(sig=record.p_signal, peak_inds=qrs_inds, fs=record.fs,
         title="GQRS peak detection on infant5ECG")
# Correct the peaks shifting them to local maxima
min_bpm = 20
max_bpm = 230
# min_gap = record.fs * 60 / min_bpm
# Use the maximum possible bpm as the search radius
search_radius = int(record.fs * 60 / max_bpm)

# corrected_peak_inds
corrected_peak_inds = processing.peaks.correct_peaks(record.p_signal[:, 0],
                                                     peak_inds=qrs_inds,
                                                     search_radius=search_radius,
                                                     smooth_window_size=150)

# Display results
# print('Corrected GQRS detected peak indices:', sorted(corrected_peak_inds)) #矫正峰值
heart_rate = peaks_hr(sig=record.p_signal, peak_inds=sorted(corrected_peak_inds), fs=record.fs,
                      title="Corrected GQRS peak detection on infant5ECG")


# resampledsignal_2d
def calculatehrin10sec(record, corrected_peak_inds):
    i = 0
    j = 0
    beatarray = []
    while i < len(record.p_signal):
        numofbeatin10sec = 0

        while corrected_peak_inds[j] < (i + ((10 / (1 / record.fs)) - 1)):
            numofbeatin10sec = numofbeatin10sec + 1
            j = j + 1
            if j >= len(corrected_peak_inds):
                break
        beatarray = beatarray + [numofbeatin10sec * 6]
        i = i + (10 / (1 / record.fs))
    return beatarray


hrarray = calculatehrin10sec(record, corrected_peak_inds)  # 我需要把 infan 1-6 变成一个datafram 7-10是另一个datafram contat 合成的意思
# print('hrarray=',hrarray)   #after contat 完之后 是normalization  如何进行： 23号，
if (len(hrarray) * 500) > len(respiration1[0]):
    a = len(respiration1[0]) % 500
    resampleArray = respiration1[0][0:(len(respiration1[0]) - a)]
    hrarray = hrarray[0:int(len(resampleArray) / 500)]
else:
    resampleArray = respiration1[0][0:(len(hrarray) * 500)]

resampledsignal_2d = resampleArray.reshape(-1, 500)
# print('resampledsignal_2d=\n', resampledsignal_2d)

df5 = pd.DataFrame(resampledsignal_2d)
# insert harray column
df5 = df5.add_prefix('respiration')
df5.insert(500, 'hrarray', hrarray)
# 显示 DataFrame
print(len(df5))
print('df5 finished----------')

'''
df6
'''

data_dir = "/Users/apple/Desktop/content/physionet.org/files/picsdb/1.0.0"
ecg_record = wfdb.rdsamp(f"{data_dir}/infant6_ecg")  # infant1、2、_ecg
plt.plot(ecg_record[0][:500])  # ECG
# plt.show()

data_dir = "/Users/apple/Desktop/content/physionet.org/files/picsdb/1.0.0"
respiration1 = wfdb.rdsamp(f"{data_dir}/infant6_resp")  # 2 、3、4
respiration1  # 呼吸 心电图

# print("resampledsignal", resampledsignal)  # resampled signal是一个个 array 组成的
ecg_record[0]

import wfdb.processing as processing

# Load the WFDB record and the physical samples
record = wfdb.rdrecord(f"{data_dir}/infant6_ecg", channels=[0])  ###   改的ecg  500->250
# Use the GQRS algorithm to detect QRS locations in the first channel
qrs_inds = processing.qrs.gqrs_detect(sig=record.p_signal[:, 0], fs=record.fs)

# # Plot results
# Load the WFDB record and the physical samples
record = wfdb.rdrecord(f"{data_dir}/infant6_ecg", channels=[0])


def findqrs(record):
    # Use the GQRS algorithm to detect QRS locations in the first channel
    qrs_inds = processing.qrs.gqrs_detect(sig=record.p_signal[:, 0], fs=record.fs)

    # Correct the peaks shifting them to local maxima
    min_bpm = 20
    max_bpm = 230
    # min_gap = record.fs * 60 / min_bpm
    # Use the maximum possible bpm as the search radius
    search_radius = int(record.fs * 60 / max_bpm)
    corrected_peak_inds = processing.peaks.correct_peaks(record.p_signal[:, 0],
                                                         peak_inds=qrs_inds,
                                                         search_radius=search_radius,
                                                         smooth_window_size=150)
    return corrected_peak_inds


corrected_peak_inds = findqrs(record)


# resampledsignal_2d
def calculatehrin10sec(record, corrected_peak_inds):
    i = 0
    j = 0
    beatarray = []
    while i < len(record.p_signal):
        numofbeatin10sec = 0

        while corrected_peak_inds[j] < (i + ((10 / (1 / record.fs)) - 1)):
            numofbeatin10sec = numofbeatin10sec + 1
            j = j + 1
            if j >= len(corrected_peak_inds):
                break
        beatarray = beatarray + [numofbeatin10sec * 6]
        i = i + (10 / (1 / record.fs))
    return beatarray


hrarray = calculatehrin10sec(record, corrected_peak_inds)  # 我需要把 infan 1-6 变成一个datafram 7-10是另一个datafram contat 合成的意思
# print('hrarray=\n', hrarray)  # after contat 完之后 是normalization  如何进行： 23号，


# respiration1[0] 表示二维数组的第一个元素
'''
取出 respiration1[0] 数组的前 len(hrarray) * 500 个元素，并赋值给 resample_array 变量/
取出了hrarray长度对应的呼吸信号数据，并赋值给resample_array.
'''

if (len(hrarray) * 500) > len(respiration1[0]):
    a = len(respiration1[0]) % 500
    resampleArray = respiration1[0][0:(len(respiration1[0]) - a)]
    hrarray = hrarray[0:int(len(resampleArray) / 500)]

else:
    resampleArray = respiration1[0][0:(len(hrarray) * 500)]


resampledsignal_2d = resampleArray.reshape(-1, 500)
# print('resampledsignal_2d=\n', resampledsignal_2d)


df6 = pd.DataFrame(resampledsignal_2d)
# insert harray column
df6 = df6.add_prefix('respiration')   # 错误1
df6.insert(500, 'hrarray', hrarray)
# 显示 DataFrame
print(len(df6),'df6 finished')

df = pd.concat([df1, df2, df3, df4, df5, df6])
outputpath = '/Users/apple/PycharmProjects/WOA7001_algorithm/Output/train.csv'
df.to_csv(outputpath, sep=',', index=False, header=True)
print(len(df))
print("contact-Finished----------")
