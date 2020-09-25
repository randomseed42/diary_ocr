import sys
import cv2
import getopt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import paddlehub as hub

def main(argv):
    img_file = ''
    output_dir = ''

    try:
        opts, args = getopt.getopt(argv, 'hi:o:', ['imgfile=', 'outputdir='])
    except getopt.GetoptError:
        print(f'single_ocr.py -i <img_file> -o <output_dir>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(f'single_ocr.py -i <img_file> -o <output_dir>')
            sys.exit()
        elif opt in ('-i', '--imgfile'):
            img_file = arg
        elif opt in ('-o', '--outputdir'):
            output_dir = arg
    
    text = diary_ocr(img_file, output_dir=output_dir)

    print(text)


def diary_ocr(img_file, output_dir='ocr_result'):
    ocr = hub.Module(name='chinese_ocr_db_crnn_server')

    imgs = [cv2.imread(img_file)]

    results = ocr.recognize_text(
        images=imgs,
        use_gpu=True,
        output_dir=output_dir,
        visualization=True,
        box_thresh=0.5,
        text_thresh=0.5,
    )
    timestamp = os.path.splitext(results[0]['save_path'])[0].split('_')[-1]

    text = '\n'.join([text['text'] for text in results[0]['data']])

    with open(os.path.join('ocr_result', f'ocrtext_{timestamp}.txt'), 'w') as f:
        f.write(text)

    return text


if __name__ == '__main__':
    main(sys.argv[1:])
