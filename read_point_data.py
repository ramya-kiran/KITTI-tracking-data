import argparse

def main():
    # Read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('source', nargs='+', help='list of image(s)')
    parser.add_argument('-o', '--output', default='result.tfrecords', help='output filename, default to result.tfrecords')

    args = parser.parse_args()

    # Convert!
    #convert(args.source, args.output)
    print(len(args.source))

if __name__ == '__main__':
    main()