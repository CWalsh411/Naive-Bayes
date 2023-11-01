import sys
import string

def lowercaseAndSeperatePunc(data):
    newstr = ""
    for x in data:
        x = x.lower()
        x = x.translate(str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))
        newstr += x
    return newstr

input = open(sys.argv[1], 'r')
sentences = input.readlines()
sentences = lowercaseAndSeperatePunc(sentences)
input.close()
output = open(sys.argv[1], 'w')
output.writelines(sentences)
output.close()
