from scipy import misc


def resize_image(image):
    ''''resizes the given image to shape 64x64 by cutting the topmost 20 and the
    bottommost 20 pixels und resizing the remaining pixels
    image is a 3-dim array (height, width, colors); returns a 3-dim array'''
    # img_orin = misc.imread(filename)
    img_square = image[20:-20]
    img_result = misc.imresize(img_square, (64, 64))
    return img_result


def resize_image_files(source, target):
    if isinstance(source, str):
        source = [source]
    source_not_found = []
    target_not_found = []
    for i in range(len(source)):
        try:
            img_orig = misc.imread(source[i])
        except FileNotFoundError:
            source_not_found.append(source[i])
        else:
            result = resize_image(img_orig)
            try:
                misc.imsave(target[i], result)
            except FileNotFoundError:
                target_not_found.append(target[i])
    if source_not_found:
        print('The following files cound not be found and were ignored:')
        print('\n'.join(source_not_found))
    if target_not_found:
        print('The following resized files cound not be written since their target folder does not exist:')
        print('\n'.join(target_not_found))


def get_path_range(start, stop, path_start, path_append):
    a= path_start+'%06d'+path_append
    return [a%i for i in range(start, stop)]

if __name__ == '__main__':
    start = 1
    stop = 10001
    source = get_path_range(start, stop, '../Datasets/img_align_celeba/', '.jpg')
    target = get_path_range(start, stop, '../Datasets/img_align_celeba_resized/', '.png')

    resize_image_files(source, target)
