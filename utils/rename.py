import os

"""

Rename directories name in WebCaricature/OriginalImages directory. 

Ex:

    "OriginalImages/Abraham Lincoln" -> "OriginalImages/Abraham_Lincoln"

"""

if __name__ == '__main__':

    prefix = '../datasets/WebCaricature/OriginalImages'

    for oldname in os.listdir(prefix):
        newname = oldname.replace(' ', '_')
        os.rename(os.path.join(prefix, oldname), os.path.join(prefix, newname))

    print("done.")