import re
import os
import numpy as np

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


def labelize_images():
    l = []
    dirpath = '/home/kasi/Documents/Deep_Learning/wiki/'
    for folder in os.listdir(dirpath):
        if folder != "wiki.mat":
            for image in os.listdir('/home/kasi/Documents/Deep_Learning/wiki/' + folder):
                image_age = find_age(image)
                # print('age on photo {}'.format(image_age))
                if image_age > 0:
                    l.append(image_age)
    vector_length = len(l)
    k = 0
    y = np.zeros((vector_length, 7))
    # print(y)
    for folder in os.listdir(dirpath):
        for image in os.listdir('/home/kasi/Documents/Deep_Learning/wiki/' + folder):
            image_age = find_age(image)
            if image_age > 0:
                y[k] = turn_age_into_vector(image_age)
                k += 1
    return y


def save_array():
    labels = labelize_images()
    np.save('wiki_labels.npy', labels)


if __name__ == '__main__':
    save_array()