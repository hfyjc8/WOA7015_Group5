import wfdb
import matplotlib.pyplot as plt

import pandas as pd

"""
infant     婴儿
frequency      频率
respiration     呼吸
corrected peak inds   校正峰值indds
"""
data_dir = "/Users/huanghefangshu/Documents/MXS_Python_Project/pythonProject/content/physionet.org/files/picsdb/1.0.0"
ecg_record = wfdb.rdsamp(f"{data_dir}/infant7_ecg")  # infant1、2、_ecg
plt.plot(ecg_record[0][:500])  # ECG
# plt.show()

data_dir = "/Users/huanghefangshu/Documents/MXS_Python_Project/pythonProject/content/physionet.org/files/picsdb/1.0.0"
respiration1 = wfdb.rdsamp(f"{data_dir}/infant7_resp")  # 2 、3、4
respiration1  # 呼吸 心电图

# resampled signal是一个个 array 组成的
ecg_record[0]

import wfdb.processing as processing

# Load the WFDB record and the physical samples
record = wfdb.rdrecord(f"{data_dir}/infant7_ecg", channels=[0])  ###   改的ecg  500->250
# Use the GQRS algorithm to detect QRS locations in the first channel
qrs_inds = processing.qrs.gqrs_detect(sig=record.p_signal[:, 0], fs=record.fs)

# # Plot results
# Load the WFDB record and the physical samples
record = wfdb.rdrecord(f"{data_dir}/infant7_ecg", channels=[0])


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
# after contat 完之后 是normalization  如何进行： 23号，
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

df7 = pd.DataFrame(resampledsignal_2d)
# insert harray column
df7 = df7.add_prefix('respiration')
df7.insert(500, 'hrarray', hrarray)
# 显示 DataFrame
print(len(df7))
'''
test8
'''

data_dir = "/Users/huanghefangshu/Documents/MXS_Python_Project/pythonProject/content/physionet.org/files/picsdb/1.0.0"
ecg_record = wfdb.rdsamp(f"{data_dir}/infant8_ecg")  # infant1、2、_ecg
plt.plot(ecg_record[0][:500])  # ECG
# plt.show()

data_dir = "/Users/huanghefangshu/Documents/MXS_Python_Project/pythonProject/content/physionet.org/files/picsdb/1.0.0"
respiration1 = wfdb.rdsamp(f"{data_dir}/infant8_resp")  # 2 、3、4
respiration1  # 呼吸 心电图
# resampled signal是一个个 array 组成的
ecg_record[0]

import wfdb.processing as processing

# Load the WFDB record and the physical samples
record = wfdb.rdrecord(f"{data_dir}/infant8_ecg", channels=[0])  ###   改的ecg  500->250
# Use the GQRS algorithm to detect QRS locations in the first channel
qrs_inds = processing.qrs.gqrs_detect(sig=record.p_signal[:, 0], fs=record.fs)

# # Plot results
# Load the WFDB record and the physical samples
record = wfdb.rdrecord(f"{data_dir}/infant8_ecg", channels=[0])


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


hrarray = calculatehrin10sec(record, corrected_peak_inds)  # 我需要把 infant 1-6 变成一个datafram 7-10是另一个datafram contat 合成的意思
# after contat 完之后 是normalization  如何进行： 23号，


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

# print('resample_array=\n',resample_array)
# print('len(resample_array)=',len(resample_array))
resampledsignal_2d = resampleArray.reshape(-1, 500)

df8 = pd.DataFrame(resampledsignal_2d)
# insert harray column
df8 = df8.add_prefix('respiration')
df8.insert(500, 'hrarray', hrarray)
# 显示 DataFrame
print(len(df8))
'''
df9
'''

data_dir = "/Users/huanghefangshu/Documents/MXS_Python_Project/pythonProject/content/physionet.org/files/picsdb/1.0.0"
ecg_record = wfdb.rdsamp(f"{data_dir}/infant9_ecg")  # infant1、2、_ecg
plt.plot(ecg_record[0][:500])  # ECG
# plt.show()


data_dir = "/Users/huanghefangshu/Documents/MXS_Python_Project/pythonProject/content/physionet.org/files/picsdb/1.0.0"
respiration1 = wfdb.rdsamp(f"{data_dir}/infant9_resp")  # 2 、3、4
respiration1  # 呼吸 心电图

# print("resampledsignal", resampledsignal)  # resampled signal是一个个 array 组成的
ecg_record[0]

import wfdb.processing as processing

# Load the WFDB record and the physical samples
record = wfdb.rdrecord(f"{data_dir}/infant9_ecg", channels=[0])  ###   改的ecg  500->250
# Use the GQRS algorithm to detect QRS locations in the first channel
qrs_inds = processing.qrs.gqrs_detect(sig=record.p_signal[:, 0], fs=record.fs)

# # Plot results
# Load the WFDB record and the physical samples
record = wfdb.rdrecord(f"{data_dir}/infant9_ecg", channels=[0])


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
# after contat 完之后 是normalization  如何进行： 23号，

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

df9 = pd.DataFrame(resampledsignal_2d)
# insert harray column
df9 = df9.add_prefix('respiration')
df9.insert(500, 'hrarray', hrarray)
# 显示 DataFrame
print(len(df9))

'''
df10
'''

data_dir = "/Users/huanghefangshu/Documents/MXS_Python_Project/pythonProject/content/physionet.org/files/picsdb/1.0.0"
ecg_record = wfdb.rdsamp(f"{data_dir}/infant10_ecg")  # infant1、2、_ecg
plt.plot(ecg_record[0][:500])  # ECG
# plt.show()

data_dir = "/Users/huanghefangshu/Documents/MXS_Python_Project/pythonProject/content/physionet.org/files/picsdb/1.0.0"
respiration1 = wfdb.rdsamp(f"{data_dir}/infant10_resp")  # 2 、3、4
respiration1  # 呼吸 心电图

ecg_record[0]
# print('ecg_record[0]', ecg_record[0])

import wfdb.processing as processing

# Load the WFDB record and the physical samples
record = wfdb.rdrecord(f"{data_dir}/infant10_ecg", channels=[0])  ###   改的ecg  500->250
# Use the GQRS algorithm to detect QRS locations in the first channel
qrs_inds = processing.qrs.gqrs_detect(sig=record.p_signal[:, 0], fs=record.fs)

# # Plot results
# Load the WFDB record and the physical samples
record = wfdb.rdrecord(f"{data_dir}/infant10_ecg", channels=[0])


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
# print('resampledsignal_2d=\n', resampledsignal_2d)

df10 = pd.DataFrame(resampledsignal_2d)
# insert harray column
df10 = df10.add_prefix('respiration')
df10.insert(500, 'hrarray', hrarray)
# 显示 DataFrame
print(len(df10))

df = pd.concat([df7, df8, df9, df10])
# output
outputpath = '/Users/huanghefangshu/Documents/MXS_Python_Project/pythonProject/Output/test.csv'
df.to_csv(outputpath, sep=',', index=False, header=False)
print(len(df))
print("拼接结束----------")
