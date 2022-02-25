from datetime import date
from hashlib import new
from matplotlib.text import OffsetFrom
import pandas as pd
import random
import numpy as np
import KC
import matplotlib.pyplot as plt
import math
from textwrap import wrap

today = date.today()
current_date = today.strftime("%b-%d-%Y")
windowSize = 15 #This is the length of a "good data" string
num_of_dots_allowed = 0 #How many dots are allowed in the "good data"
years_skipped = 1
step = 1 #How many data points we skip (granulate the data)

# file_paths_list = ["/Users/rafiqkamal/Desktop/Data_Science/timeSeries/timeSerierEnv/Brownian_Motion_Data"]#? BROWNIAN ONLY
file_paths_list = ["/Users/rafiqkamal/Desktop/Data_Science/timeSeries/timeSerierEnv/World_Health_Data/Data_Extract_From_Health_Nutrition_and_Population_Statistics.xlsx"]

# file_paths_list = ["/Users/rafiqkamal/Desktop/Data_Science/timeSeries/timeSerierEnv/World_Health_Data/Data_Extract_From_Health_Nutrition_and_Population_Statistics.xlsx","/Users/rafiqkamal/Desktop/Data_Science/timeSeries/timeSerierEnv/World_Development_Data/Data_Extract_From_World_Development_Indicators.xlsx","/Users/rafiqkamal/Desktop/Data_Science/timeSeries/timeSerierEnv/Nitrous_Oxide_Data/Data_Extract_From_World_Development_Indicators.xlsx","/Users/rafiqkamal/Desktop/Data_Science/timeSeries/timeSerierEnv/Jobs_Data/Data_Extract_From_Jobs.xlsx","/Users/rafiqkamal/Desktop/Data_Science/timeSeries/timeSerierEnv/Global_Economics_Data/Data_Extract_From_Global_Economic_Monitor_(GEM).xlsx","/Users/rafiqkamal/Desktop/Data_Science/timeSeries/timeSerierEnv/Energy_Data/Energy and Mining 50 series 200 plus entities World Bank.xlsx","/Users/rafiqkamal/Desktop/Data_Science/timeSeries/timeSerierEnv/Education_Statistics/Data_Extract_From_Education_Statistics_-_All_Indicators.xlsx","/Users/rafiqkamal/Desktop/Data_Science/timeSeries/timeSerierEnv/All_Data_Combined/Combined Data Jan-03-2022.xlsx"]#! Uncomment

def combine_csv_data(OneDArrayOfPathNames):
    files = [pd.ExcelFile(name) for name in OneDArrayOfPathNames]
    frames = [x.parse(x.sheet_names[0],header=None,index_col=None) for x in files]
    frames[1:] = [df[1:] for df in frames[1:]]
    combined = pd.concat(frames)
    combined.to_excel("{1}Combined Data {0}.xlsx".format(current_date,save_path),header=False,index=False)

# combine_csv_data(file_paths_list)#! Uncomment to make a combined excel file

def get_clean_data(arr):
    result = []
    for row in arr:
        series = []
        for item in row:
            if (type(item) == int or type(item) == float):
                series.append(item)
            if (item == '..'):
                series.append(".")
        result.append(series)
        series = []
    return result

def insert_to_random_index(array, characters, no_of_reps):#Just for test data
    for _ in range(no_of_reps):
        array = list(array)
        array.insert(random.randint(0, len(array)), characters)
    return array

def traverseTwoDArray (twoDArr):#Just for test data
    result = []
    for row in twoDArr:
        temp = insert_to_random_index(row,"..",2)
        result.append(temp)
    return result# Modifies array in place

def convertToBinary(arr):#find the difference between each value, and if it is > 0 then we write '1' and otherwise we write '0'
    result = []
    for row in arr:
        temp = []
        limit = len(row) - 1
        for i, x in enumerate(row):
            if i == limit:
                continue
            diff = 0 if x - row[i + 1] > 0 else 1
            temp.append(diff)
        result.append(temp)
    return result

def get_convertToBinary_String(list):
    temp = []
    limit = len(list) - 1
    for i, x in enumerate(list):
        if i == limit:
            continue
        diff = 0 if x - list[i + 1] > 0 else 1
        temp.append(diff)
    return "".join([str(x) for x in temp])

# Series value becomes 1 if value is >mean of the series, 0 otherwise
def MoreLessMean(X):
    binary_series = []
    for row in X:
        row_ = row - np.mean(row)
        b = ''.join([ str(int(r>0)) for r in row_])
        binary_series.append(b)
    return binary_series

def get_MoreLessMean_String_Of_list(X):
    row_ = X - np.mean(X)
    b = ''.join([ str(int(r>0)) for r in row_])
    return b

# Series value becomes 1 if value is >median of the series, 0 otherwise
def MoreLessMedian(X):
    binary_series = []
    for row in X:
        row_ = row - np.median(row)
        b = ''.join([ str(int(r>0)) for r in row_])
        binary_series.append(b)
    return binary_series

# normalise the series to be between 0 and 1, then set "1" id >0.5
def MoreLessHalf(X):
    binary_series = []
    for row in X:
        row_ = row - np.min(row)
        row_ = row_/np.max(row_)# row_ has values between 0 and 1 now
        b = ''.join([ str(int(r>0.5)) for r in row_])
        binary_series.append(b)
    return binary_series

# subtract the linear trend
def LinearDetrendDiff(X):
    binary_series = []
    for row in X:
        t = np.arange(len(row))
        fit = np.polyfit(t,row,1)# fit linear func y = mx+b to series
        row_ = row - (fit[1]*t + fit[0])
        b = ''.join([str(int(val>0)) for val in np.diff(row_)])
        binary_series.append(b)
    return binary_series

def get_string_of_LinearDetrendDiff(list):
    t = np.arange(len(list))
    fit = np.polyfit(t,list,1)# fit linear func y = mx+b to series
    row_ = list - (fit[1] * t + fit[0])
    b = ''.join([str(int(val>0)) for val in np.diff(row_)])
    return b

# subtract the linear trend
def AboveLinearDetrend(X):
    binary_series = []
    for row in X:
        t = np.arange(len(row))
        fit = np.polyfit(t,row,1)# fit linear func y = mx+b to series
        row_ = row - (fit[1]*t + fit[0])
        b = ''.join([ str(int(r>0)) for r in row_])
        binary_series.append(b)
    return binary_series

def get_string_AboveLinearDetrend(list):
    t = np.arange(len(list))
    fit = np.polyfit(t,list,1)# fit linear func y = mx+b to series
    row_ = list - (fit[1]*t + fit[0])
    b = ''.join([ str(int(r>0)) for r in row_])
    return b

def K_scaled_all_series(listOfTuples):
    result = []

    for tuple in listOfTuples:
        result.append(tuple[2])
    return result

def is_good_data(list):
    for element in list:
        if isinstance(element, str) or np.isnan(float(element)) or not isinstance(element, (int,float)):
            return False
    return True

def sliding_window(listArray,window_size):
    reversed_array = list(reversed(listArray))
    for i in range(len(reversed_array) - window_size + 1):
        temp = list(reversed_array[i:i + window_size + 1])
        if is_good_data(temp):
            return list(reversed(temp))
    return False

def has_string(list):
    for x in list:
        if type(x) == str:
            return True

def get_good_string(arr):#search each row for most recent instance of good data
    result = []
    for row in arr:
        good_list = sliding_window(row,windowSize )
        if good_list:
            result.append(good_list)
        else:
            continue
    return result

def join_list_into_string(npArray):
    result = []
    for row in npArray:
        temp = "".join(map(str,row))
        result.append(temp)
    return result

def get_frequencies(twoD_array):
    return dict((x,twoD_array.count(x)) for x in set(twoD_array))

def sort_dictionary(dict):
    return sorted(dict.items(),key=lambda x:x)

def get_frequencies_list(twoDList):
    result = []
    for tuple in twoDList:
        result.append(tuple[1])
    return result

def get_complexities(list):
    result = []
    for i, x in enumerate(list):
        k = KC.calc_KC(x[0])
        k = np.round(k,1)
        result.append(k)
        list[i] = x + (k,)
    return result
                            #  string      freq  complx
#sortedDict mutated=[... ('101111111111111', 48, 13.7), ('111111111111111', 895, 3.9)]

def get_complexities_2D_list(twoDList):
    result = []
    for i,row in enumerate(twoDList):
        k = KC.calc_KC(row)
        k = np.round(k,1)
        result.append(k)
        twoDList[i] = row + (k,)
    return result

def get_log10(list):
    newArray = np.array(list)
    result = []
    for x in newArray:
        result.append(np.log10(x))
    return result

def get_probabilities(list):
    result = []
    N = sum(list)
    for num in list:
        temp = num / N
        result.append(temp)
    return result

def get_Up_Bound(list):
    newList = np.array(list)
    Up_Bound = 2**-newList
    return Up_Bound

def get_kscaled(list):
    K = np.array(list)
    # a_set = set(K)
    # N = len(a_set)
    N = 2**windowSize
    K_scaled = np.log2(N) * (K - np.min(K)) / (np.max(K) - np.min(K))
    return K_scaled

def get_min_max_freq_patterns(listArray):
    result = []
    for tuple in listArray:
        temp = []
        temp2 = []
        complexity = tuple[0]
        pattern_of_max = tuple[1][0][0]
        pattern_of_min = tuple[1][1][0]
        max_freq = tuple[1][0][1]
        min_freq = tuple[1][1][1]
        temp.append(complexity)
        temp.append(pattern_of_max)
        temp.append(max_freq)
        temp2.append(complexity)
        temp2.append(pattern_of_min)
        temp2.append(min_freq)
        result.append(temp)
        result.append(temp2)
    return result

def make_plot(list1,list2,function,with_annotations=False):#plots np.log10(frequency) vs complexity
    withBinary = "with_Binary" if with_annotations else "without_Binary"
    probability = get_probabilities(list1)
    scaledComplexities = np.round(get_kscaled(list2),1)#make upper from complex
    upperBound = get_Up_Bound(scaledComplexities)
    log10List = np.round(get_log10(probability),3)
    normalListOfFrequencies = np.round(probability,3)
    upperBound = np.log10(upperBound)

    prediction_success_rate = PredictWhichIsHigherProb(probability,list2)
    subscription = ""
    tild = "{K}"

    if function.__name__ == "LinearDetrendDiff":
        subscription = "{Det}"
    elif function.__name__ == "MoreLessMean":
        subscription = "{HL}"
    elif function.__name__ == "convertToBinary":
        subscription = "{UD}"

    temp = []
    for i,row in enumerate(scaledComplexities):
        temp.append([sortedDict[i][0],normalListOfFrequencies[i],scaledComplexities[i]])

    unique_complexities = get_unique_complexities(temp)

    temp2 = []
    for i,row in enumerate(scaledComplexities):
        temp2.append([sortedDict[i][0],log10List[i],scaledComplexities[i]])

    log10_unique_complexities = get_unique_complexities(temp2)

    font = {#make a copy of the things that will change
        "size":28,
        "weight":"bold",
    }
    # plt.rc("font",**font)
    plt.plot(scaledComplexities, upperBound, "-", color="black", label=f'Success Rate: {prediction_success_rate}')
    plt.plot(scaledComplexities,log10List, linestyle="" ,marker="o",color="blue", ms=12)
    plt.gcf().set_size_inches(13, 9)
    plt.ylim(( min(log10List) - 0.25 ,0 ))
    plt.xlabel(r'$\tilde{K}_{HL}(x)$',fontsize=label_fontsize) #! UNCOMMENT
    plt.ylabel(r'$\log_{10}P(x)$',fontsize=label_fontsize)
    ax = plt.gca()
    ax.tick_params(axis="both", which="both", labelsize=22)

    # print(length_of_series)
    # plt.title("15_Chars_Brownian_Motion_Number_of_series:{2}_{4}".format(windowSize,years_skipped,length_of_series,data_short_name,withBinary),fontsize=label_fontsize)# ? UNCOMMENT for Brownian Run
    # plt.xlabel("Complexities [Binarise Method: {0}]".format(function.__name__),fontsize=label_fontsize)# ? UNCOMMENT for Brownian Run
    # plt.ylabel(r'$\log_{10}P(x)$',fontsize=label_fontsize)

    unique_complexities.sort(key=lambda x: x[0])

    list_of_pattern_complex_freq = get_min_max_freq_patterns(unique_complexities)
    #Returns list of [pattern,complexity,max,min]

    string_unique_complexities = get_strings_of_tuples(list_of_pattern_complex_freq)#! uncomment

    log10_unique_complexities.sort(key=lambda x: x[0])

    if windowSize == 20:
        every_other_complexity = log10_unique_complexities[::3]
        rotation_degrees = 60
    elif windowSize >= 10:
        every_other_complexity = log10_unique_complexities[::2]
        rotation_degrees = 45
    else:
        every_other_complexity = log10_unique_complexities[::2]
        rotation_degrees = 0

    del every_other_complexity[-2]

    i = 0
    if with_annotations:
        for tuple in every_other_complexity:
            complexity = tuple[0]
            pattern_of_max = tuple[1][0][0]
            pattern_of_min = tuple[1][1][0]
            max_freq = tuple[1][0][1]
            min_freq = tuple[1][1][1]

            if i == 0:
                offset = -70
                minOffset = 50
                yMinOffset = -95
                minHa = "left"
            elif i == len(every_other_complexity) - 1:
                offset = 140
            else:
                offset = -20
                minOffset = -25
                yMinOffset = -25
                minHa = "right"

            if i == len(every_other_complexity) - 1:
                xMaxOffset = -45
                yMinOffset = -40
                maxHa = "center"
            else:
                xMaxOffset = 15
                maxHa = "left"

            text1 = plt.annotate(
                pattern_of_max,
                xy=(complexity, max_freq), xytext=(complexity + xMaxOffset, max_freq + offset),
                textcoords='offset points', ha=maxHa, va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=1),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3',),
                annotation_clip=False,rotation=rotation_degrees
            )
            text2 = plt.annotate(
                pattern_of_min,
                xy=(complexity, min_freq), xytext=(complexity + minOffset, min_freq + yMinOffset),
                textcoords='offset points', ha=minHa, va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='0.9', alpha=1),
                arrowprops=dict(arrowstyle='->', connectionstyle='angle3,angleA=0,angleB=90'),
                annotation_clip=False,rotation=rotation_degrees
            )
            i += 1

            text1.set_fontsize(label_fontsize)
            text2.set_fontsize(label_fontsize)

    all_data_labels = []
    labels = []
    x_array = []
    y_array = []
    for tuple in sortedDict:
        all_data_labels.append(tuple[0])

    for x,y in every_other_complexity:#unique_complexities
        x_array.append(x)
        y_array.append(y[0][1])
        labels.append(y[0][0])

    for x,y in every_other_complexity:#unique_complexities
        x_array.append(x)
        y_array.append(y[1][1])
        labels.append(y[1][0])

    ##############[['11111111111111111111', 0.0, -0.382, -1.791]
    # with open("{4}15_Chars_{1}_Pattern_Complexity_Max_and_Min_Brownian_Motion_{2}.txt".format(windowSize,func.__name__,current_date,data_short_name,save_path), 'w') as f:
        # f.write(string_unique_complexities)# ? UNCOMMENT for Brownian Run

    # plt.savefig('{4}Scatter_Plot_15_Chars_{1}_Brownian_Motion_{6}_{3}'.format(windowSize,func.__name__,data_short_name,current_date,save_path,years_skipped,withBinary),bbox_inches='tight')# ? UNCOMMENT for Brownian Run

    # with open("{4}{0}Chars_{1}_Pattern_Complexity_Max_and_Min_{3}_{2}.txt".format(windowSize,func.__name__,current_date,data_short_name,save_path), 'w') as f: #! uncomment
    #     f.write(string_unique_complexities)#! uncomment

    # if with_annotations:
        # with open("{4}Prediction_Success_Rate_{0}_Chars_{1}_{3}_{2}.txt".format(windowSize,func.__name__,current_date,data_short_name,save_path), 'w') as f: #! uncomment
                # f.write("".join(str(prediction_success_rate)))#! uncomment

    # with open("{4}Prediction_Success_Rate_15_Chars_{1}_Brownian_Motion_{2}.txt".format(windowSize,func.__name__,current_date,data_short_name,save_path), 'w') as f:
        # f.write(str(prediction_success_rate))# ? UNCOMMENT for Brownian Run

    plt.savefig('{4}Scatter_Plot_{0}_Chars_{1}_{2}_{3}_{5}'.format(windowSize,func.__name__,data_short_name,current_date,save_path,withBinary))#! Uncomment

    plt.show()#! Uncomment

def make_histogram(listOfAllComplexities):
    plt.hist(listOfAllComplexities,ec='black',color='orange',log=True,density=True,bins=5)
    plt.xlabel(r'$\tilde{K}_{HL}(x)$',fontsize=label_fontsize)
    plt.ylabel('Frequency',fontsize=label_fontsize)
    ax = plt.gca()
    ax.tick_params(axis="both", which="both", labelsize=19)

    # plt.title('Hist_15_Chars_{1}_Brownian_Motion'.format(windowSize,func.__name__,data_short_name))# ? UNCOMMENT for Brownian Run
    # plt.savefig('{4}Hist_15_Chars_{1}_Brownian_Motion_{3}'.format(windowSize,func.__name__,data_short_name,current_date,save_path,years_skipped))# ? UNCOMMENT for Brownian Run

    # plt.title('Hist_{0}_Chars_{1}_{2}'.format(windowSize,func.__name__,data_short_name),fontsize=label_fontsize)

    plt.savefig('{4}Hist_{0}_Chars_{1}_{2}_{3}'.format(windowSize,func.__name__,data_short_name,current_date,save_path,years_skipped),bbox_inches='tight')#! Uncomment
    plt.show()

def add_complexities_to_string(listOfFrequencies,listOfComplexities):
    result = []
    for i, x in enumerate(listOfFrequencies):
        temp = x + (listOfComplexities[i],)
        result.append(temp)
    return result

def get_strings_of_tuples(twoDArray):
    result = []
    for x in twoDArray:
        temp = " ".join(map(str,x))
        result.append(temp)
    result = "\n".join(result)
    return result

def get_unique_complexities(listOfTuples):
    seen = {}
    for listArray in listOfTuples:
        currComplexity = listArray[2]
        if currComplexity not in seen:
            #  complexity key           max                min
            seen[currComplexity] = [[listArray[0],listArray[1]],[listArray[0],listArray[1]]]
        else:
            seen[currComplexity][0] = [listArray[0],listArray[1]] if seen[currComplexity][0][1] < listArray[1] else seen[currComplexity][0]
            seen[currComplexity][1] = [listArray[0],listArray[1]] if seen[currComplexity][1][1] > listArray[1] else seen[currComplexity][1]
    return list(seen.items())

def PredictWhichIsHigherProb(P,K):
    assert(len(P)==len(K))
    assert(np.abs(sum(P)-1)<0.001)

    success_rate = []
    for samps in range(10000):# do samples to test our predictions
        # pick a random series x and series y, according to their probabilities, and record their probabilities
        indx = np.random.choice(np.arange(len(K)),p=P)
        indy = np.random.choice(np.arange(len(K)),p=P)
        if K[indx] < K[indy]:
            #predict x is more likely that y (or equal)
            success_rate.append(1*(P[indx]>=P[indy]))
        elif K[indy] < K[indx]:
            success_rate.append(1*(P[indy]>=P[indx]))
        elif K[indy] == K[indx]:
            #if the complexities are the same, flip a coin to predict which has higher probability
            success_rate.append(1*(np.random.rand()>0.5))

    return np.sum(success_rate)/len(success_rate)

dataInfo = []

def get_All_Numbers_From_twoDArray(twoDArray):
    result = []
    i = 0
    for row in twoDArray:
        temp = []
        for element in row:
            if isinstance(element, str) or np.isnan(float(element)) or math.isnan(element) or not isinstance(element, (int,float)) or math.isinf(element):
                continue
            else:
                temp.append(element)
        if (len(temp) > 20):
            dataInfo.append(row[0:5])
            result.append(i)
            result.append(temp)
        i += 1
    return result

def TESTgetAllNumbers(list):
    temp = []
    for element in list:
        if isinstance(element, str) or np.isnan(float(element)) or math.isnan(element) or not isinstance(element, (int,float)) or math.isinf(element):
            continue
        else:
            temp.append(element)
    return temp

def get_standard_deviation(list):
    return np.std(list)

def get_most_volatile_list(twoDArray):
    max = 0
    maxList = []

    for row in twoDArray:
        currStd = get_standard_deviation(row) / (sum(row) / len(row))
        if max < currStd:
            max = currStd
            maxList = row
    return maxList

def BrownianMotion(years=15,record=1):
    #Brownian motion for (10) years, recorded every "record" years
    Gaussian = np.random.randn(years)
    Bm = np.cumsum(Gaussian)# Brownian motion
    if record==1:
        SaveBm = Bm
    elif record!=1:
         SaveBm = [Bm[j] for j in range(len(Bm)) if (j%record)==0]
    return SaveBm

def getBrownianMotionSeries():
    result = []
    for i in range(10000):
        series = BrownianMotion(years=15,record=1)
        result.append(series)
    return result

brownian_motion_series = getBrownianMotionSeries()

for file in file_paths_list:
    fileArray = file.split("/")
    fileName = file
    data_short_name = fileArray[len(fileArray) - 2]#! UNCOMMENT
    save_path = "/".join(fileArray[:-1]) + "/" #!UNCOMMENT
    # data_short_name = fileArray[len(fileArray) - 1] #? BROWNIAN ONLY
    save_path = "/".join(fileArray) + "/" #? BROWNIAN ONLY

    print("file name: ",fileName,"\ndata short name: ",data_short_name,"\nsave path: ",save_path)

# fileName = "/Users/rafiqkamal/Desktop/Data_Science/timeSeries/timeSerierEnv/Energy_Data/Energy and Mining 50 series 200 plus entities World Bank.xlsx"#insert file name here
# save_path = "/Users/rafiqkamal/Desktop/Data_Science/timeSeries/timeSerierEnv/Energy_Data/"#Put desired file path here
# data_short_name = "Energy Data" #! Change with new data set

    originalData = pd.read_excel(fileName) #! UNCOMMENT
    df = originalData.copy()#! UNCOMMENT
    label_fontsize = 25 #! UNCOMMENT

    # CONTENTS:
    #1. remove labels and non key data
    # 2. delete columns
    # 3. sliding window to get good strings
    # 4. convert np.array to binary
    # 5. make plots

    #! Test Suite: uncomment these and comment the line below to see some test data
    ###1. remove labels and non key data
    # test = np.array(random.sample(range(100),100)).reshape(10,10)
    # print(test)
    # test = traverseTwoDArray(test)# Modifies array in place and adds dots
    # print("\n Added some '..' to the data to simulate the real data \n", test,"\n\n")

    # #### # 2. sliding window to get good strings
    # goodStringsArray = get_good_string(test)
    # print("Got the good strings out of the array\n",np.array(goodStringsArray),"\n\n")

    # ### # 3. delete columns
    # newDF = pd.DataFrame(goodStringsArray)
    # print("converted list to a data frame\n",newDF,"\n\n")
    # newDF = newDF.iloc[:,::years_skipped]
    # print("delete every other row\n",newDF,"\n\n")
    # newDF = newDF.to_numpy()
    # print("convert the data frame to a numpy array\n",newDF,"\n\n")

    # ### # 4. convert np.array to binary
    # testResult = get_good_string(newDF)
    # print("getting all the binary strings from the np array\n",np.array(testResult),"Number of time series: ",len(testResult))

    arrayOfArrays = df.to_numpy(copy=True)#! UNCOMMENT

    binarise_array = [LinearDetrendDiff,MoreLessMean,convertToBinary]#! LinearDetrendDiff,MoreLessMean,convertToBinary
    # ,AboveLinearDetrend,MoreLessHalf,MoreLessMedian

    for func in binarise_array:#! Uncomment
        for x in [15]:#! Uncomment 5,10,11,15,20
            for y in [1]:#! Uncomment
                years_skipped = y#! Uncomment
                windowSize = x#! Uncomment

################################################################################
################################################################################
################################################################################
# Creating Visualization Charts for LinearDetrend and MoreLessMean of Denmark series Data: Adults (ages 15+) and children (0-14 years) living with HIV
                # newArrayOfArryas = get_All_Numbers_From_twoDArray(arrayOfArrays)  # ? UNCOMMENT for Brownian Run
                # newArrayOfArryas = get_good_string(arrayOfArrays)
                # volatile_list_of_lists = []
                # years_labels = np.arange(1971,2021) #? VISUALIZATION
                # print("List of strings", volatile_binary_strings_list_of_lists)
                # print(newArrayOfArryas)

                # newArrayOfArryas = np.array(newArrayOfArryas)
                # i = 0
                # for row in newArrayOfArryas:
                #     volatile_list = np.array(row)
                #     MoreLessMean_binary_string = get_MoreLessMean_String_Of_list(volatile_list)
                #     volatile_binary_strings_linear = get_string_of_LinearDetrendDiff(volatile_list)

                #     convertToBinaryString = get_convertToBinary_String(row)

                #     print(i, "More:",MoreLessMean_binary_string)
                #     # print(i, "Linear",volatile_binary_strings_linear)
                #     # print(i, "Convert",convertToBinaryString)
                #     i += 1


                # print("number of rows: ", len(newArrayOfArryas))
                # print(newArrayOfArryas.shape)
                # min = 1000
                # tempArray = []
                # lengthOfRow = 20

                # for i,row in enumerate(newArrayOfArryas):
                #     if len(row) >= lengthOfRow:
                #         tempArray.append(row[:lengthOfRow])

                # resultOfLinear = LinearDetrendDiff(tempArray)

                # for i,row in enumerate(resultOfLinear):
                #     row = str(i) row

                # print(resultOfLinear)
                # print(len(resultOfLinear))
                # uniformArray = [np.array(a) for a in newArrayOfArryas]
                # uniformArray = np.array(uniformArray)
                # print(uniformArray.shape)

                # binaryStringsOfMethod = LinearDetrendDiff(newArrayOfArryas)
                # print(binaryStringsOfMethod)

                # for idx in [180,194,786,792,804,808,815,818,834,839,878]: #[1,66,83]:#24582,24853,24824,24673,24741,24700,24630,24614,24615,24582,22307,19849,7906,7794,7733,7798,7393,7387,537
                #     volatile_list = np.array(newArrayOfArryas[idx])
                #     volatile_binary_strings = get_string_of_LinearDetrendDiff(volatile_list)

                #     i = 0
                #     plt.figure()
                #     for q in volatile_list:
                #         if i >= len(volatile_list) - 2:
                #             break
                #         if volatile_binary_strings[i]=='0':
                #             col = 'b'
                #         else:
                #             col = 'orange'
                #         plt.plot([i,i+1],[volatile_list[i],volatile_list[i+1]],color=col,lw=3)
                #         i = i+1

                #     plt.title("{1}".format(windowSize,data_short_name))
                #     plt.xlabel("Years [Binarise Method: LinearDetrend]".format(func.__name__))
                #     plt.ylabel("Population")
                #     plt.show()
    # #!These are coming out as all zeros and some have variation in MoreLess np.arange(400,1000)
    # ? UNCOMMENT for VISLUALIZATION Run
                # indexesAndArrays = newArrayOfArryas
                # indexes = newArrayOfArryas[::2]
                # newArrayOfArryas = newArrayOfArryas[1::2]

                # for binarise in [get_convertToBinary_String]:#180,834,878
                #     for idx in [180,834,878]:#24582,24853,24824,24673,24741,24700,24630,24614,24615,24582,22307,19849,7906,7794,7733,7798,7393,7387,537

                #         if idx == 834:
                #             functionCall = get_convertToBinary_String
                #         elif idx == 180:
                #             functionCall = get_MoreLessMean_String_Of_list
                #         else:
                #             functionCall = get_string_of_LinearDetrendDiff

                #         volatile_list = np.array(newArrayOfArryas[idx])
                #         binaryString = functionCall(volatile_list)
                #         dataTitleLong = ""

                #         if idx == 180:
                #             yLabel = "New HIV Infections" #Germany
                #             dataTitleLong = "Adults Newly Infected with HIV" #Germany
                #         else:
                #             yLabel = "% of Coverage for Women\nwith HIV in"
                #             dataTitleLong = "Percentage of Antiretroviral Therapy Coverage for PMTCT for Pregnant Women Living with HIV in"

                #         sizeOfLabels = 30
                #         print(indexes[idx + 2],idx,binaryString,newArrayOfArryas[idx],dataInfo[idx])

                #         i = 0

                #         plt.figure(figsize = (13, 11), dpi = 80)
                #         for q in volatile_list:
                #             if i >= len(volatile_list) - 2:
                #                 break
                #             if binaryString[i]=='0':
                #                 col = 'b'
                #             else:
                #                 col = 'orange'
                #             plt.plot([1990 + i,1990 + i+1],[volatile_list[i],volatile_list[i+1]],color=col,lw=3)
                #             i = i+1

                #         plt.xlabel("Years".format(func.__name__,indexes[idx]),fontsize=sizeOfLabels)
                #         plt.ylabel("{1} {0}".format(dataInfo[idx][1],yLabel),fontsize=sizeOfLabels)
                #         plt.yticks(fontsize=sizeOfLabels)
                #         plt.xticks(np.arange(1990,2020,4), fontsize=sizeOfLabels)
                #         plt.xlim(1990,1990 + len(newArrayOfArryas[idx]))
                #         plt.savefig('{0}Visualization_{7}_{6}_{2}_{8}_{5}_{4}_{1}_{3}'.format(save_path,binaryString,functionCall.__name__,current_date,data_short_name,yLabel.replace("\n"," "),dataInfo[idx][1],idx,dataTitleLong),format='png', bbox_inches='tight')
                #         plt.show()

    # ? UNCOMMENT for VISLUALIZATION Run
################################################################################Visualization_878_Yemen, Rep._get_string_of_LinearDetrendDiff_Percentage of Antiretroviral Therapy Coverage for PMTCT for Pregnant Women Living with HIV in_World_Health_Data_00000010110110101010_Feb-23-2022
################################################################################
################################################################################

                #! remove labels and non key data: delete columns
                #! sliding window to get good strings
                test = get_good_string(arrayOfArrays)# Finds strings with all numbers #! Uncomment
                # # print("Cleaned data:\n", test,"\n\n","Number of serier",len(test))

                newDF = (pd.DataFrame(test))#! uncomment
                # # print("converted list to a data frame\n",newDF,"\n\n")\

                skippedDataArray = newDF.iloc[:,::years_skipped]#! uncomment #Remove every other column to make data more coarse

                new = skippedDataArray.to_numpy()#! uncomment
                # print("convert the data frame to a numpy array\n",new[0],"\n\n",type(new))
                # goodStringsArray = get_good_string(test)

                # print("Got the good strings out of the array\n",goodStringsArray,"\n\n",goodStringsArray[0] ,"Number of series: ", len(goodStringsArray))

                # stringArray = np.array(testResult)
                # print(stringArray.dtype.type is str)
                # print(get_good_string(testResult))#This shows that it is now checking for strings too

                # testResult = func(brownian_motion_series)# ? UNCOMMENT for Brownian Run

                # #! convert to binary
                testResult = func(new)#! uncomment
                # # # print(testResult, "getting all the binary strings from the np array\n","\nNumber of time series: ",len(testResult))
                # # # # print(testResult[0])
                # # # # Make list of strings of binary

                listOfbinaryStrings = join_list_into_string(testResult)#! uncomment
                # # print(listOfbinaryStrings,"\nNumber of series: ",len(listOfbinaryStrings))

                # # #! Get frequencies and complexities
                frequencies = get_frequencies(listOfbinaryStrings)#! uncomment
                # print(frequencies)

                # # #! Sort frequencies
                sortedDict = dict(sorted(frequencies.items(),key=lambda item: item[1]))#! uncomment
                sortedDict = list(sortedDict.items())#! uncomment
                # # print(sortedDict)#Mutated
                #                                 #  string      freq  complx
                # # sortedDict mutated=[... ('101111111111111', 48, 13.7), ('111111111111111', 895, 3.9)]
                # # print("Length of the sorted Dictionary" ,len(sortedDict))

                # # ! get complexities
                complexities = get_complexities(sortedDict)#[53.1,41.1,35.1,...]
                # #! uncomment

                list_of_all_series_complexities = K_scaled_all_series(sortedDict)#! uncomment

                # # # ! sort object of frequencies
                frequency_list = get_frequencies_list(sortedDict)#[...1,...59, 62, 288, 1370]#! uncomment
                length_of_series = sum(frequency_list)#! uncomment
                # print("sum of frequencies",length_of_series,"Chars:",x,func.__name__)
                # print("Number of frequencies: ",len(frequency_list))
                # print( "Number of Series:", len(list_of_all_series_complexities))
                # print("length of frequency list: ", len(frequencies))
                # # print(np.mean(list_of_all_series_complexities))
                # # print(np.std(list_of_all_series_complexities))

                # # ! 5. make plots and histogram
                make_plot(frequency_list,complexities,func,False)#! uncomment
                # make_plot(frequency_list,complexities,func,True)#! uncomment
                # make_histogram(list_of_all_series_complexities)#! uncomment
                # print(list_of_all_series_complexities)

                # ! 6. Save results in a text file
                final_string_log10freq_complexity = get_strings_of_tuples(sortedDict)#! uncomment
                # 000000000000000 1632 3.9\n000000000000000 1632 3.9...
                # print(final_string_log10freq_complexity)

                # with open("{3}{0}Chars_{1}_Pattern_Freq_Complexity_{2}.txt".format(windowSize,func.__name__,current_date,save_path), 'w') as f: #! uncomment
                #     f.write(final_string_log10freq_complexity)#! uncomment

                # with open("{4}Prediction_Success_Rate_{0}_Chars_{1}_{3}_{2}.txt".format(windowSize,func.__name__,current_date,data_short_name,save_path), 'w') as f: #! uncomment
                #     f.write("".join(map(str,prediction_success_rate_list)))#! uncomment

                # with open("{3}15_Chars_{1}_Brownian_Motion_Pattern_Freq_Complexity_{2}.txt".format(windowSize,func.__name__,current_date,save_path), 'w') as f:
                    # f.write(final_string_log10freq_complexity)# ? UNCOMMENT for Brownian Run

                # with open("{4}Prediction_Success_Rate_15_Chars_{1}_Brownian_Motion_{2}.txt".format(windowSize,func.__name__,current_date,data_short_name,save_path), 'w') as f:
                #     f.write("".join(map(str,prediction_success_rate_list)))# ? UNCOMMENT for Brownian Run