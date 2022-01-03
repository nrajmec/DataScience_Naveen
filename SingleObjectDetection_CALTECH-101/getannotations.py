import scipy.io
import os
import cv2


def get_annotations():
    csv_path = './dataset/airplanes.csv'
    annotations_path = "./Annotations/Airplanes_Side_2"
    images_path = './dataset/images'

    allfiles = os.listdir(annotations_path)


    lst_annots = []
    for file in allfiles:
        # if file == 'annotation_0002.mat':
        mat = scipy.io.loadmat(os.path.join(annotations_path, file))

        print(mat['box_coord'][0])

        (y1, y2, x1, x2) = mat['box_coord'][0]

        imagename = file.replace('annotation', 'image')
        imagename = imagename.replace('.mat', '.jpg')

        img = cv2.imread(os.path.join(images_path, imagename))

        # color = (255, 0, 0)
        # thickness = 1
        # cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        #
        # cv2.imshow('Image', img)
        # cv2.waitKey(0)
        str_coords = ','.join(str(ele) for ele in (x1, y1, x2, y2))
        lst_annots.append(imagename + ',' + str_coords)

    writeannotations = open(csv_path, "w+")
    writeannotations.write('\n'.join(lst_annots))
    writeannotations.close()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    get_annotations()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
