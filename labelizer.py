import re
import os
import numpy as np

DIRPATH = '/home/kasi/Documents/Deep_Learning/wiki/'


def find_age(str):
    regex = r"_([0-9]+).+_([0-9]+)"
    matches = re.search(regex, str)
    if matches:
        birth = matches.group(1)
        picture = matches.group(2)
        age = int(picture) - int(birth)
        # print("Year of birth : {}".format(birth))
        # print("Year of picture : ".format(picture))
        # print("Age : {}".format(age))
        return age
# find_age('23300_1962-06-19_2011.jpg')


def turn_age_into_vector(age):
    if   age < 20 :
        return np.array([1, 0, 0, 0, 0, 0, 0])
    elif age >= 20 and age < 30:
        return np.array([0, 1, 0, 0, 0, 0, 0])
    elif age >= 30 and age < 40:
        return np.array([0, 0, 1, 0, 0, 0, 0])
    elif age >= 40 and age < 50:
        return np.array([0, 0, 0, 1, 0, 0, 0])
    elif age >= 50 and age < 60:
        return np.array([0, 0, 0, 0, 1, 0, 0])
    elif age >= 60 and age < 70:
        return np.array([0, 0, 0, 0, 0, 1, 0])
    elif age >=70:
        return np.array([0, 0, 0, 0, 0, 0, 1])
    else:
        pass


def labelize_images(n):
    folders = range(n)
    l = []
    for folder in folders:
        for image in os.listdir('/home/kasi/Documents/Deep_Learning/wiki/' + str(folder).zfill(2)):
            image_age = find_age(image)
            # print('age on photo {}'.format(image_age))
            if image_age > 0:
                l.append(image_age)
    vector_length = len(l)
    k = 0
    y = np.zeros((vector_length, 7))
    # y = np.zeros((575, 7))
    counter = 0
    for folder in folders:
        print(folder)
        for image in os.listdir('/home/kasi/Documents/Deep_Learning/wiki/' + str(folder).zfill(2)):
            image_age = find_age(image)
            if image_age > 0:
                y[k] = turn_age_into_vector(image_age)
                counter += 1
                k += 1
    print("number of pictures stored in training set: {}".format(counter))
    return y


def save_array(n):
    labels = labelize_images(n)
    np.save('wiki_labels.npy', labels)



if __name__ == "__main__":
    number_of_folders = 20
    save_array(number_of_folders)

# f= np.load('wiki_labels.npy')
# print(f)