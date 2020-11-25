from Logger import log
import numpy as np
import pandas as pd


class ChunkDoubleSourceSlider(object):
    def __init__(self, filename, batchsize, chunksize, shuffle, offset, crop=None, header=0, ram_threshold=5*10**5):

        self.filename = filename
        self.batchsize = batchsize
        self.chunksize = chunksize
        self.shuffle = shuffle
        self.offset = offset
        self.header = header
        self.crop = crop
        self.ram = ram_threshold

    def check_lenght(self):
        # check the csv size
        check_cvs = pd.read_csv(self.filename,
                                nrows=self.crop,
                                chunksize=10 ** 3,
                                header=self.header
                                )

        t_size = 0

        for chunk in check_cvs:
            size = chunk.shape[0]
            t_size += size
            del chunk
        log('Size of the dataset is {:.3f} M rows.'.format(t_size/10 ** 6))
        if t_size > self.ram:  # IF dataset is too large for memory
            log('It is too large to fit in memory so it will be loaded in chunkes of size {:}.'.format(self.chunksize))
        else:
            log('This size can fit the memory so it will load entirely')

        return t_size

    def feed_chunk(self):

        try:
            total_size
        except NameError:
            #global total_size
            total_size = ChunkDoubleSourceSlider.check_lenght(self)

        if total_size > self.ram:  # IF dataset is too large for memory

            # LOAD data from csv
            data_frame = pd.read_csv(self.filename,
                                     nrows=self.crop,
                                     chunksize=self.chunksize,
                                     header=self.header
                                     )

            # iterations over csv file
            for chunk in data_frame:

                np_array = np.array(chunk)
                inputs, targets = np_array[:, 0], np_array[:, 1]

                """
                if len(inputs) < self.batchsize:
                    while len(inputs) == self.batchsize:
                        inputs = np.append(inputs, 0)
                       targets = np.append(targets, 0)
                """

                max_batchsize = inputs.size - 2 * self.offset
                if self.batchsize < 0:
                    self.batchsize = max_batchsize

                # define indices and shuffle them if necessary
                indices = np.arange(max_batchsize)
                if self.shuffle:
                    np.random.shuffle(indices)

                # providing sliding windows:
                for start_idx in range(0, max_batchsize, self.batchsize):

                    excerpt = indices[start_idx:start_idx + self.batchsize]

                    inp = np.array([inputs[idx:idx + 2 * self.offset + 1] for idx in excerpt])
                    tar = targets[excerpt + self.offset].reshape(-1, 1)

                    yield inp, tar

        else:  # IF dataset can fit the memory

            # LOAD data from csv
            data_frame = pd.read_csv(self.filename,
                                     nrows=self.crop,
                                     header=self.header
                                     )

            np_array = np.array(data_frame)
            inputs, targets = np_array[:, 0], np_array[:, 1]


            max_batchsize = inputs.size - 2 * self.offset
            if self.batchsize < 0:
                    self.batchsize = max_batchsize

            # define indices and shuffle them if necessary
            indices = np.arange(max_batchsize)
            if self.shuffle:
                    np.random.shuffle(indices)

            # providing sliding windows:
            for start_idx in range(0, max_batchsize, self.batchsize):
                excerpt = indices[start_idx:start_idx + self.batchsize]

                inp = np.array([inputs[idx:idx + 2 * self.offset + 1] for idx in excerpt])
                tar = targets[excerpt + self.offset].reshape(-1, 1)

                yield inp, tar


class ChunkDoubleSourceSlider2(object):
    def __init__(self, filename, batchsize, chunksize, shuffle, offset, crop=None, header=0, ram_threshold=5 * 10 ** 5):

        self.filename = filename
        self.batchsize = batchsize
        self.chunksize = chunksize
        self.shuffle = shuffle
        self.offset = offset
        self.header = header
        self.crop = crop
        self.ram = ram_threshold
        self.total_size = 0

    def check_length(self):
        # check the csv size
        check_cvs = pd.read_csv(self.filename,
                                nrows=self.crop,
                                chunksize=10 ** 3,
                                header=self.header
                                )

        for chunk in check_cvs:
            size = chunk.shape[0]
            self.total_size += size
            del chunk
        log('Size of the dataset is {:.3f} M rows.'.format(self.total_size / 10 ** 6))
        if self.total_size > self.ram:  # IF dataset is too large for memory
            log('It is too large to fit in memory so it will be loaded in chunkes of size {:}.'.format(self.chunksize))
        else:
            log('This size can fit the memory so it will load entirely')

    def feed_chunk(self):

        if self.total_size == 0:
            ChunkDoubleSourceSlider2.check_length(self)

        if self.total_size > self.ram:  # IF dataset is too large for memory

            # LOAD data from csv
            data_frame = pd.read_csv(self.filename,
                                     nrows=self.crop,
                                     chunksize=self.chunksize,
                                     header=self.header
                                     )

            skip_idx = np.arange(self.total_size/self.chunksize)
            if self.shuffle:
                np.random.shuffle(skip_idx)

            log(str(skip_idx), 'debug')

            for i in skip_idx:

                log('index: ' + str(i), 'debug')

                # Read the data
                data = pd.read_csv(self.filename,
                                   nrows=self.chunksize,
                                   skiprows=int(i)*self.chunksize,
                                   header=self.header)

                np_array = np.array(data)
                inputs, targets = np_array[:, 0], np_array[:, 1]

                max_batchsize = inputs.size - 2 * self.offset
                if self.batchsize < 0:
                    self.batchsize = max_batchsize

                # define indices and shuffle them if necessary
                indices = np.arange(max_batchsize)
                if self.shuffle:
                    np.random.shuffle(indices)

                # providing sliding windows:
                for start_idx in range(0, max_batchsize, self.batchsize):
                    excerpt = indices[start_idx:start_idx + self.batchsize]

                    inp = np.array([inputs[idx:idx + 2 * self.offset + 1] for idx in excerpt])
                    tar = targets[excerpt + self.offset].reshape(-1, 1)

                    yield inp, tar

        else:  # IF dataset can fit the memory

            # LOAD data from csv
            data_frame = pd.read_csv(self.filename,
                                     nrows=self.crop,
                                     header=self.header
                                     )

            np_array = np.array(data_frame)
            inputs, targets = np_array[:, 0], np_array[:, 1]

            max_batchsize = inputs.size - 2 * self.offset
            if self.batchsize < 0:
                self.batchsize = max_batchsize

            # define indices and shuffle them if necessary
            indices = np.arange(max_batchsize)
            if self.shuffle:
                np.random.shuffle(indices)

            # providing sliding windows:
            for start_idx in range(0, max_batchsize, self.batchsize):
                excerpt = indices[start_idx:start_idx + self.batchsize]

                inp = np.array([inputs[idx:idx + 2 * self.offset + 1] for idx in excerpt])
                tar = targets[excerpt + self.offset].reshape(-1, 1)

                yield inp, tar


class ChunkDoubleSourceSlider2_online(object):
    def __init__(self, filename, batchsize, chunksize, shuffle, offset, crop=None, header=0, ram_threshold=5 * 10 ** 5):

        self.filename = filename
        self.batchsize = batchsize
        self.chunksize = chunksize
        self.shuffle = shuffle
        self.offset = offset
        self.header = header
        self.crop = crop
        self.ram = ram_threshold
        self.total_size = 0

    def check_length(self):
        # check the csv size
        check_cvs = pd.read_csv(self.filename,
                                nrows=self.crop,
                                chunksize=10 ** 3,
                                header=self.header
                                )

        for chunk in check_cvs:
            size = chunk.shape[0]
            self.total_size += size
            del chunk
        log('Size of the dataset is {:.3f} M rows.'.format(self.total_size / 10 ** 6))
        if self.total_size > self.ram:  # IF dataset is too large for memory
            log('It is too large to fit in memory so it will be loaded in chunkes of size {:}.'.format(self.chunksize))
        else:
            log('This size can fit the memory so it will load entirely')

    def feed_chunk(self):

        if self.total_size == 0:
            ChunkDoubleSourceSlider2_online.check_length(self)

        if self.total_size > self.ram:  # IF dataset is too large for memory

            # LOAD data from csv
            data_frame = pd.read_csv(self.filename,
                                     nrows=self.crop,
                                     chunksize=self.chunksize,
                                     header=self.header
                                     )

            skip_idx = np.arange(self.total_size/self.chunksize)
            if self.shuffle:
                np.random.shuffle(skip_idx)

            log(str(skip_idx), 'debug')

            for i in skip_idx:

                log('index: ' + str(i), 'debug')

                # Read the data
                data = pd.read_csv(self.filename,
                                   nrows=self.chunksize,
                                   skiprows=int(i)*self.chunksize,
                                   header=self.header)

                np_array = np.array(data)
                inputs, targets = np_array[:, 0], np_array[:, 1]

                max_batchsize = inputs.size - 2 * self.offset
                if self.batchsize < 0:
                    self.batchsize = max_batchsize

                # define indices and shuffle them if necessary
                indices = np.arange(max_batchsize)
                if self.shuffle:
                    np.random.shuffle(indices)

                # providing sliding windows:
                for start_idx in range(0, max_batchsize, self.batchsize):
                    excerpt = indices[start_idx:start_idx + self.batchsize]

                    inp = np.array([inputs[idx:idx + 2 * self.offset + 1] for idx in excerpt])
                    tar = targets[excerpt + self.offset * 2].reshape(-1, 1)

                    yield inp, tar

        else:  # IF dataset can fit the memory

            # LOAD data from csv
            data_frame = pd.read_csv(self.filename,
                                     nrows=self.crop,
                                     header=self.header
                                     )

            np_array = np.array(data_frame)
            inputs, targets = np_array[:, 0], np_array[:, 1]

            max_batchsize = inputs.size - 2 * self.offset
            if self.batchsize < 0:
                self.batchsize = max_batchsize

            # define indices and shuffle them if necessary
            indices = np.arange(max_batchsize)
            if self.shuffle:
                np.random.shuffle(indices)

            # providing sliding windows:
            for start_idx in range(0, max_batchsize, self.batchsize):
                excerpt = indices[start_idx:start_idx + self.batchsize]

                inp = np.array([inputs[idx:idx + 2 * self.offset + 1] for idx in excerpt])
                tar = targets[excerpt + self.offset * 2].reshape(-1, 1)

                yield inp, tar




class ChunkDoubleSourceSlider2_even(object):
    def __init__(self, filename, batchsize, chunksize, shuffle, offset, crop=None, header=0, ram_threshold=5 * 10 ** 5):

        self.filename = filename
        self.batchsize = batchsize
        self.chunksize = chunksize
        self.shuffle = shuffle
        self.offset = offset
        self.header = header
        self.crop = crop
        self.ram = ram_threshold
        self.total_size = 0

    def check_length(self):
        # check the csv size
        check_cvs = pd.read_csv(self.filename,
                                nrows=self.crop,
                                chunksize=10 ** 3,
                                header=self.header
                                )

        for chunk in check_cvs:
            size = chunk.shape[0]
            self.total_size += size
            del chunk
        log('Size of the dataset is {:.3f} M rows.'.format(self.total_size / 10 ** 6))
        if self.total_size > self.ram:  # IF dataset is too large for memory
            log('It is too large to fit in memory so it will be loaded in chunkes of size {:}.'.format(self.chunksize))
        else:
            log('This size can fit the memory so it will load entirely')

    def feed_chunk(self):

        if self.total_size == 0:
            ChunkDoubleSourceSlider2_even.check_length(self)

        if self.total_size > self.ram:  # IF dataset is too large for memory

            # LOAD data from csv
            data_frame = pd.read_csv(self.filename,
                                     nrows=self.crop,
                                     chunksize=self.chunksize,
                                     header=self.header
                                     )

            skip_idx = np.arange(self.total_size/self.chunksize)
            if self.shuffle:
                np.random.shuffle(skip_idx)

            log(str(skip_idx), 'debug')

            for i in skip_idx:

                log('index: ' + str(i), 'debug')

                # Read the data
                data = pd.read_csv(self.filename,
                                   nrows=self.chunksize,
                                   skiprows=int(i)*self.chunksize,
                                   header=self.header)

                np_array = np.array(data)
                inputs, targets = np_array[:, 0], np_array[:, 1]

                max_batchsize = inputs.size - 2 * self.offset
                if self.batchsize < 0:
                    self.batchsize = max_batchsize

                # define indices and shuffle them if necessary
                indices = np.arange(max_batchsize)
                if self.shuffle:
                    np.random.shuffle(indices)

                # providing sliding windows:
                for start_idx in range(0, max_batchsize, self.batchsize):
                    excerpt = indices[start_idx:start_idx + self.batchsize]

                    inp = np.array([inputs[idx:idx + 2 * self.offset] for idx in excerpt])
                    tar = targets[excerpt + 2 * self.offset-1].reshape(-1, 1)

                    yield inp, tar

        else:  # IF dataset can fit the memory

            # LOAD data from csv
            data_frame = pd.read_csv(self.filename,
                                     nrows=self.crop,
                                     header=self.header
                                     )

            np_array = np.array(data_frame)
            inputs, targets = np_array[:, 0], np_array[:, 1]

            max_batchsize = inputs.size - 2 * self.offset
            if self.batchsize < 0:
                self.batchsize = max_batchsize

            # define indices and shuffle them if necessary
            indices = np.arange(max_batchsize)
            if self.shuffle:
                np.random.shuffle(indices)

            # providing sliding windows:
            for start_idx in range(0, max_batchsize, self.batchsize):
                excerpt = indices[start_idx:start_idx + self.batchsize]

                inp = np.array([inputs[idx:idx + 2 * self.offset] for idx in excerpt])
                tar = targets[excerpt + 2* self.offset -1].reshape(-1, 1)

                yield inp, tar

class DoubleSourceProvider2(object):

    def __init__(self, batchsize, shuffle, offset):

        self.batchsize = batchsize
        self.shuffle = shuffle
        self.offset = offset

    def feed(self, inputs, targets):

        assert len(inputs) == len(targets)

        inputs = inputs.flatten()
        targets = targets.flatten()

        max_batchsize = inputs.size - 2 * self.offset

        if self.batchsize == -1:
            self.batchsize = len(inputs)

        indices = np.arange(max_batchsize)
        if self.shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, max_batchsize, self.batchsize):
            excerpt = indices[start_idx:start_idx + self.batchsize]

            yield np.array([inputs[idx:idx + 2 * self.offset + 1] for idx in excerpt]),\
                  targets[excerpt + self.offset].reshape(-1, 1)


class DoubleSourceProvider3(object):

    def __init__(self, nofWindows, offset):

        self.nofWindows = nofWindows
        self.offset = offset

    def feed(self, inputs):

        inputs = inputs.flatten()
        ###########
        # inputs = np.pad(inputs,(self.offset,self.offset),'constant',constant_values=(0,0))
        ############
        max_nofw = inputs.size - 2 * self.offset

        if self.nofWindows < 0:
            self.nofWindows = max_nofw

        indices = np.arange(max_nofw, dtype=int)

        # providing sliding windows:
        for start_idx in range(0, max_nofw, self.nofWindows):
            excerpt = indices[start_idx:start_idx + self.nofWindows]

            inp = np.array([inputs[idx:idx + 2 * self.offset + 1] for idx in excerpt])

            yield inp

class DoubleSourceProvider3_even(object):

    def __init__(self, nofWindows, offset):

        self.nofWindows = nofWindows
        self.offset = offset

    def feed(self, inputs):

        inputs = inputs.flatten()
        ###########
        # inputs = np.pad(inputs,(self.offset,self.offset),'constant',constant_values=(0,0))
        ############
        max_nofw = inputs.size - 2 * self.offset +1

        if self.nofWindows < 0:
            self.nofWindows = max_nofw

        indices = np.arange(max_nofw, dtype=int)

        # providing sliding windows:
        for start_idx in range(0, max_nofw, self.nofWindows):
            excerpt = indices[start_idx:start_idx + self.nofWindows]

            inp = np.array([inputs[idx:idx + 2 * self.offset] for idx in excerpt])

            yield inp

class DoubleSourceProvider4(object):

    def __init__(self, nofWindows, offset, windowlength):

        self.nofWindows = nofWindows
        self.offset = offset
        self.windowlength = windowlength

    def feed(self, inputs):

        inputs = inputs.flatten()
        # max_nofw = inputs.size - self.windowlength
        max_nofw = inputs.size - self.windowlength + 1
        if self.nofWindows < 0:
            self.nofWindows = max_nofw

        indices = np.arange(max_nofw, dtype=int)

        # providing sliding windows:
        for start_idx in range(0, max_nofw, self.nofWindows):
            excerpt = indices[start_idx:start_idx + self.nofWindows]

            yield np.array([inputs[idx:idx + self.windowlength] for idx in excerpt])
            #tar = np.array([inputs[idx:idx + self.windowlength] for idx in excerpt])

class DoubleSourceProvider4_online(object):

    def __init__(self, nofWindows, offset, windowlength):

        self.nofWindows = nofWindows
        self.offset = offset
        self.windowlength = windowlength

    def feed(self, inputs):

        inputs = inputs.flatten()
        # max_nofw = inputs.size - self.windowlength
        max_nofw = inputs.size - self.windowlength + 1
        if self.nofWindows < 0:
            self.nofWindows = max_nofw

        indices = np.arange(max_nofw, dtype=int)

        # providing sliding windows:
        for start_idx in range(0, max_nofw, self.nofWindows):
            excerpt = indices[start_idx:start_idx + self.nofWindows]

            yield np.array([inputs[idx:idx + self.windowlength] for idx in excerpt])
            #tar = np.array([inputs[idx:idx + self.windowlength] for idx in excerpt])

class DoubleSourceProvider_fcn(object):

    def __init__(self, nofWindows, offset, windowlength):

        self.nofWindows = nofWindows
        self.offset = offset
        self.windowlength = windowlength

    def feed(self, inputs):

        inputs = inputs.flatten()
        output_length = self.windowlength
        num_windows = int(np.ceil((inputs.size - 2 * self.offset) / output_length))
        # pad the end of the dataframe so that it has an exact multiple of windows
        # pad_size = num_windows * output_length + 2 * self.offset - inputs.size
        # pad_zeros = np.zeros(pad_size)
        # inputs = np.concatenate((inputs, pad_zeros), )
        max_nofw = num_windows
        input_length = 2 * self.offset + output_length
        if self.nofWindows < 0:
            self.nofWindows = max_nofw

        indices = np.arange(max_nofw, dtype=int)

        # providing sliding windows:
        for start_idx in range(0, max_nofw, self.nofWindows):
            excerpt = indices[start_idx:start_idx + self.nofWindows]

            inp = np.array([inputs[idx * output_length:idx * output_length + input_length] for idx in excerpt])
            #tar = np.array([inputs[idx:idx + self.windowlength] for idx in excerpt])

            yield inp
            #tar = np.array([inputs[idx:idx + self.windowlength] for idx in excerpt])


class MultiApp_Slider(object):

    def __init__(self, batchsize, shuffle, offset):

        self.batchsize = batchsize
        self.shuffle = shuffle
        self.offset = offset

    def feed(self, inputs, targets, flatten=True):

        # inputs, targets = inputs.flatten(), targets.flatten()
        inputs = inputs.flatten()

        assert inputs.shape[0] == targets.shape[0]
        max_batchsize = inputs.size - 2 * self.offset
        if self.batchsize < 0:
            self.batchsize = max_batchsize

        indices = np.arange(max_batchsize)
        if self.shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, max_batchsize, self.batchsize):
            excerpt = indices[start_idx:start_idx + self.batchsize]
            # if flatten:
            #     yield np.array([inputs[idx:idx + 2 * self.offset + 1] for idx in excerpt]), \
            #           targets[excerpt + self.offset]
            # else:
            yield np.array([inputs[idx:idx + 2 * self.offset + 1] for idx in excerpt]), \
                  targets[excerpt + self.offset, :]


class ChunkS2S_Slider_fcn(object):
    def __init__(self, filename, batchsize, chunksize, shuffle, length, crop=None, header=0, ram_threshold=5 * 10 ** 5):

        self.filename = filename
        self.batchsize = batchsize
        self.chunksize = chunksize
        self.shuffle = shuffle
        self.length = length
        self.header = header
        self.crop = crop
        self.ram = ram_threshold
        self.total_size = 0

    def check_length(self):
        # check the csv size
        check_cvs = pd.read_csv(self.filename,
                                nrows=self.crop,
                                chunksize=10 ** 3,
                                header=self.header
                                )

        for chunk in check_cvs:
            size = chunk.shape[0]
            self.total_size += size
            del chunk
        log('Size of the dataset is {:.3f} M rows.'.format(self.total_size / 10 ** 6))
        if self.total_size > self.ram:  # IF dataset is too large for memory
            log('It is too large to fit in memory so it will be loaded in chunkes of size {:}.'.format(self.chunksize))
        else:
            log('This size can fit the memory so it will load entirely')

    def feed_chunk(self):

        if self.total_size == 0:
            ChunkS2S_Slider_fcn.check_length(self)

        if self.total_size > self.ram:  # IF dataset is too large for memory

            # LOAD data from csv
            data_frame = pd.read_csv(self.filename,
                                     nrows=self.crop,
                                     chunksize=self.chunksize,
                                     header=self.header
                                     )

            skip_idx = np.arange(self.total_size/self.chunksize)
            if self.shuffle:
                np.random.shuffle(skip_idx)

            log(str(skip_idx), 'debug')

            for i in skip_idx:

                log('index: ' + str(i), 'debug')

                # Read the data
                data = pd.read_csv(self.filename,
                                   nrows=self.chunksize,
                                   skiprows=int(i)*self.chunksize,
                                   header=self.header)

                np_array = np.array(data)
                inputs, targets = np_array[:, 0], np_array[:, 1]

                offset = self.length // 2
                output_length = self.length
                num_windows = int(np.ceil((inputs.size - 2 * offset) / output_length))
                # pad the end of the dataframe so that it has an exact multiple of windows
                pad_size = num_windows * output_length + 2 * offset - inputs.size
                pad_zeros = np.zeros(pad_size)
                inputs = np.concatenate((inputs, pad_zeros), )
                input_length = 2 * offset + output_length
                targets = np.concatenate((targets, pad_zeros), )
                targets = targets[offset:-offset]

                max_batchsize = num_windows

                # max_batchsize = inputs.size - self.length + 1
                if self.batchsize < 0:
                    self.batchsize = max_batchsize

                # define indices and shuffle them if necessary
                indices = np.arange(max_batchsize)
                if self.shuffle:
                    np.random.shuffle(indices)

                # providing sliding windows:
                for start_idx in range(0, max_batchsize, self.batchsize):
                    excerpt = indices[start_idx:start_idx + self.batchsize]

                    inp = np.array([inputs[idx * output_length:idx * output_length + input_length] for idx in excerpt])
                    tar = np.array([targets[idx * output_length:(idx + 1) * output_length] for idx in excerpt])

                    yield inp, tar

        else:  # IF dataset can fit the memory

            # LOAD data from csv
            data_frame = pd.read_csv(self.filename,
                                     nrows=self.crop,
                                     header=self.header
                                     )

            np_array = np.array(data_frame)
            inputs, targets = np_array[:, 0], np_array[:, 1]

            offset = self.length // 2
            output_length = self.length
            num_windows = int(np.ceil((inputs.size - 2 * offset) / output_length))
            # pad the end of the dataframe so that it has an exact multiple of windows
            pad_size = num_windows * output_length + 2 * offset - inputs.size
            pad_zeros = np.zeros(pad_size)
            inputs = np.concatenate((inputs,pad_zeros),)
            input_length = 2 * offset + output_length
            # inputs = np.vstack([inputs[i*output_length:i* output_length + input_length] for i in range(num_windows)])

            targets = np.concatenate((targets, pad_zeros), )
            targets = targets[offset:-offset]

            max_batchsize = num_windows
            if self.batchsize < 0:
                self.batchsize = max_batchsize

            # define indices and shuffle them if necessary
            indices = np.arange(max_batchsize)
            if self.shuffle:
                np.random.shuffle(indices)

            # providing sliding windows:
            for start_idx in range(0, max_batchsize, self.batchsize):
                excerpt = indices[start_idx:start_idx + self.batchsize]

                inp = np.array([inputs[idx * output_length:idx * output_length + input_length] for idx in excerpt])
                tar = np.array([targets[idx*output_length:(idx+1)*output_length] for idx in excerpt])

                yield inp, tar

            # targets = targets.reshape(num_windows,output_length)
            # for i in range(0,num_windows,self.batchsize):
            #     inp = inputs[i*self.batchsize:(i+1)*self.batchsize]
            #     tar = targets[i*self.batchsize:(i+1)*self.batchsize]
            #     yield inp, tar



class ChunkS2S_Slider(object):
    def __init__(self, filename, batchsize, chunksize, shuffle, length, crop=None, header=0, ram_threshold=5 * 10 ** 5):

        self.filename = filename
        self.batchsize = batchsize
        self.chunksize = chunksize
        self.shuffle = shuffle
        self.length = length
        self.header = header
        self.crop = crop
        self.ram = ram_threshold
        self.total_size = 0

    def check_length(self):
        # check the csv size
        check_cvs = pd.read_csv(self.filename,
                                nrows=self.crop,
                                chunksize=10 ** 3,
                                header=self.header
                                )

        for chunk in check_cvs:
            size = chunk.shape[0]
            self.total_size += size
            del chunk
        log('Size of the dataset is {:.3f} M rows.'.format(self.total_size / 10 ** 6))
        if self.total_size > self.ram:  # IF dataset is too large for memory
            log('It is too large to fit in memory so it will be loaded in chunkes of size {:}.'.format(self.chunksize))
        else:
            log('This size can fit the memory so it will load entirely')

    def feed_chunk(self):

        if self.total_size == 0:
            ChunkS2S_Slider.check_length(self)

        if self.total_size > self.ram:  # IF dataset is too large for memory

            # LOAD data from csv
            data_frame = pd.read_csv(self.filename,
                                     nrows=self.crop,
                                     chunksize=self.chunksize,
                                     header=self.header
                                     )

            skip_idx = np.arange(self.total_size/self.chunksize)
            if self.shuffle:
                np.random.shuffle(skip_idx)

            log(str(skip_idx), 'debug')

            for i in skip_idx:

                log('index: ' + str(i), 'debug')

                # Read the data
                data = pd.read_csv(self.filename,
                                   nrows=self.chunksize,
                                   skiprows=int(i)*self.chunksize,
                                   header=self.header)

                np_array = np.array(data)
                inputs, targets = np_array[:, 0], np_array[:, 1]

                max_batchsize = inputs.size - self.length + 1
                if self.batchsize < 0:
                    self.batchsize = max_batchsize

                # define indices and shuffle them if necessary
                indices = np.arange(max_batchsize)
                if self.shuffle:
                    np.random.shuffle(indices)

                # providing sliding windows:
                i = 0
                for start_idx in range(0, max_batchsize, self.batchsize):
                    excerpt = indices[start_idx:start_idx + self.batchsize]

                    inp = np.array([inputs[idx:idx + self.length] for idx in excerpt])
                    tar = np.array([targets[idx:idx + self.length] for idx in excerpt])

                    yield inp, tar

        else:  # IF dataset can fit the memory

            # LOAD data from csv
            data_frame = pd.read_csv(self.filename,
                                     nrows=self.crop,
                                     header=self.header
                                     )

            np_array = np.array(data_frame)
            inputs, targets = np_array[:, 0], np_array[:, 1]

            max_batchsize = inputs.size - self.length + 1
            if self.batchsize < 0:
                self.batchsize = max_batchsize

            # define indices and shuffle them if necessary
            indices = np.arange(max_batchsize)
            if self.shuffle:
                np.random.shuffle(indices)

            # providing sliding windows:
            for start_idx in range(0, max_batchsize, self.batchsize):
                excerpt = indices[start_idx:start_idx + self.batchsize]

                inp = np.array([inputs[idx:idx + self.length] for idx in excerpt])
                tar = np.array([targets[idx:idx + self.length] for idx in excerpt])

                yield inp, tar