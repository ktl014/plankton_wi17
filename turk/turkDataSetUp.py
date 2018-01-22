'''
writeWeblinks.py

Created on Nov 20 2017 15:54 
#@author: Kevin Le 
'''
import os
import glob
import pandas as pd

'''
Create URLs for images by replacing with server dir with web dir and write text file 
with new image paths
'''

#SERVER_DIR = '/data4/plankton_wi17/plankton/for_turk/turk_images/lab'
SERVER_DIR = '/data4/plankton_wi17/plankton/images_orig/20170120/001/0000000_static_html'
WEB_DIR = 'http://www.svcl.ucsd.edu/~morgado/plankton_turk'

def createImgUrl(imgList, filename):
    '''
    Create image url
    :param imgList: list of images
    :param turkpath: 
    :return: 
    '''
    assert isinstance(imgList, list)
    assert isinstance(filename, str)
    with open ("{}.txt".format (filename), "w") as f:
        for img in imgList:
            img = img.replace (SERVER_DIR, WEB_DIR)
            f.write (img + '\n')
        f.close ()
'''
# Multiple textfiles of image paths to replace dir
SRC = '/data4/plankton_wi17/plankton/for_turk/turk_images/lab/annotationSample'
turkList = sorted(glob.glob(SRC + '/*'))
for turk in turkList:
    imgList = sorted(glob.glob(turk + '/*'))
'''
# Multiple textfiles of image paths to replace dir

# turk = 'test'
# imgList = sorted (glob.glob (turk + '/*'))
# imgList = ['/data4/plankton_wi17/plankton/images_orig/20170120/001/0000000_static_html/spcdata.html']
# createImgUrl(imgList, turk)
def writeImgPaths(imgList, filename, destpath):
    with open ('{}/{}.txt'.format (destpath,filename), 'w') as txtFile:
        txtFile.write('\n'.join(imgList))
    txtFile.close()
    print filename + ' textfile written'

def grabGoogleImgList():
    src = '/data4/plankton_wi17/plankton/Google_dataset/filtered/tempforClasses'    # SRC path for Google images
    dest = '/data4/plankton_wi17/plankton/for_turk/turk_images/google'
    specimenList = sorted(glob.glob(src + '/*'))
    for specimen in specimenList:
        imgList = sorted(glob.glob(specimen + '/*'))
        filename = os.path.basename(specimen)
        writeImgPaths(imgList, filename, dest)

def extractData():
    taxonomyBranch = ['family']
    csvFNames = ['/data4/plankton_wi17/plankton/plankton_family/{}_timestamp.csv'.format(i) for i in taxonomyBranch]  # Add file attachment to taxonomy
    for i, csvF in enumerate(csvFNames):
        df = pd.read_csv(csvF)

        # Format and clean up data
        df = df.dropna(how='any')   # Drop NaN values
        df_timeStamp = df['Timestamp: '].str.replace("-","\r")
        df_timeStamp = df_timeStamp.str.split('\r')
        df['MLClass'] = ['class{:02}'.format(i) for i in range(len(df))]

        # Convert to list
        specimenTimestamp = df_timeStamp.tolist()       # List of timestamps
        specimenClass = df['class'].tolist()  # List of classes
        specimenNum = df['# of Specimen:'].tolist()     # List of specimens per class
        mlClass = df['MLClass'].tolist()

        specimenTimestamp = [filter(None, i) for i in specimenTimestamp]    # Clean up empty values from split
        if verifyN_TStampVsSpecimen(specimenTimestamp,specimenNum):         # Check if Tstamp and specimens match up
            # datasetDict = dict(zip(mlClass, specimenTimestamp))
            datasetDict = dict (zip (specimenClass, specimenTimestamp))

        return datasetDict

def verifyN_TStampVsSpecimen(tstamp,spec):
    if len(spec) == len(tstamp):
        print ("Timestamp and Specimen count verified")
        return True
    else:
        return False

def grabLabImgList():
    plankClassDict = extractData ()
    destpath= '/data4/plankton_wi17/plankton/for_turk/turk_images/lab/phase2_annotation'
    for taxon_class, taxon_subclass in plankClassDict.items():
        for idx, timestamp_fn in enumerate(taxon_subclass):
            specimen_path = os.path.join('/data4/plankton_sp17/image_orig', timestamp_fn)
            imgList = glob.glob(os.path.join(specimen_path,"*"))
            if len(imgList) == 0:
                raise "EmptyList"
            filename = taxon_class.split(' ')[0] + '_specimen{0:02d}'.format(idx)
            writeImgPaths(imgList, filename, destpath)

def getSpecimenlinks(fileName):
    '''
    Read image urls from text file and return as a list 
    '''
    with open(fileName, 'r') as f:
        readData = f.readlines()
    f.close()
    readData = [data.strip('\n') for data in readData]
    return readData

def cleanList(srcList, tstList):
    path = '/'.join(srcList[0].split('/')[:5])
    srcList = [i.split('/')[6] for i in srcList]
    map(lambda x:srcList.remove(x), tstList)
    srcList = [os.path.join(path, i) for i in srcList]
    return srcList

def cleanList_tst():
    classList = ['/data4/plankton_wi17/plankton/for_turk/turk_images/lab/ClassA',
                 '/data4/plankton_wi17/plankton/for_turk/turk_images/lab/ClassB',
                 '/data4/plankton_wi17/plankton/for_turk/turk_images/lab/ClassC']
    specimenList = [sorted(glob.glob(specimenDir + '/*')) for specimenDir in classList]
    tstpath = '/data4/plankton_wi17/plankton/for_turk/turk_images/lab/phase2_annotation'
    tst_specimenList = sorted(glob.glob(tstpath + '/*'))
    destpath = '/data4/plankton_wi17/plankton/for_turk/turk_images/lab/phase2_poselabeling'
    for tstspecimen in tst_specimenList:
        tstname = os.path.basename(tstspecimen).split('.')[0]
        for specimen in specimenList:
            for spec in specimen:
                srcname = os.path.basename(spec)
                if srcname == tstname:
                    # tstList = glob.glob(tstspecimen + '/*')
                    # tstList = [img.split('/')[9] for img in tstList]

                    tstList = getSpecimenlinks(tstspecimen)
                    srcList = glob.glob(spec + '/*')
                    srcList = [img.split('/')[9] for img in srcList]

                    # srcList = getSpecimenlinks (spec)
                    new_srcList = cleanList (tstList, srcList)
                    print '{} images in tst, {} images after clean, {}'.format(len(tstList), len(new_srcList), tstname)
                    writeImgPaths(new_srcList, tstname, destpath)

def main():
    # Grab google images
    # grabGoogleImgList()
    # grabLabImgList()
    cleanList_tst()

if __name__ == "__main__":
    main()

