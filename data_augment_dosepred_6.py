import numpy as np
import threading

class crop_dataGenerator (object):
    """
    Function
    ----------
    This function takes a specified image path and returns n number of randomly
        generated images of the given size and depth.

    Parameters
    ----------
    patient_data: np.array
        An array of patient data
    labels: np.array
        An array of labels corresponding to the patient data array
    normalized: bool (Default: True)
        Determines if the images are normalized by group
    dimensions: tuple (Default = (256, 256, 3))
        Declares the deminsions of the windowed images
    z_location: int (Default = 1)
        The location of the training label in the stack of n_images
    n_sets: int
        Determines the number of image groups to be returned by the function

    Attributes
    ----------
    _x_max: int (Default = 512)
        The maximum size of x-axis on the input images in patient_data
    _y_max: int (Default = 512)
        The maximum size of y-axis on the input images in patient_data

    Returns
    ----------
    generator: a generator object
        Returns a generator object which produces n_images specified croped images
            when called
    """


    def __init__ (self, patient_data, labels, normalized = True,
                  dimensions = (256,256,3), z_location = 1, n_sets = 1000,
                  channels_last = True, 
                  single_patient_batch = False, 
                  single_slice_batch = False, 
                  Dim2_option = False, 
                  **kwargs):
        self.patient_data = patient_data
        self.labels = labels
        self.dimensions = dimensions
        self.normalized = normalized
        self.z_location = z_location
        self.n_sets = n_sets
        self._patient_index = []
        self.channels_last = channels_last
        self._x_max = 512
        self._y_max = 512
        self.single_patient_batch = single_patient_batch
        self.single_slice_batch = single_slice_batch
        self.Dim2_option = Dim2_option

    def _gen_index(self, type, z_max = None):
        """
        Function
        ---------
        Creates a random integer between 0 and the axis limits

        Returns
        ---------
        Returns the created integer
        """
        if type == 'x':
            if not(self._x_max-self.dimensions[0]):
                return(0)
            else:
                return(np.random.randint(low = 0, high = self._x_max - self.dimensions[0]))
        elif type == 'y':
            if not(self._y_max-self.dimensions[1]):
                return(0)
            else:
                return(np.random.randint(low = 0, high = self._y_max - self.dimensions[1]))
        elif type == 'z':
            if not(z_max-self.dimensions[2]):
                return(0)
            else:
                return(np.random.randint(low = 0, high = z_max - self.dimensions[2]))
                            

    def _xyz_generator (self, patient_index):
        """
        Function
        ----------
        Randomly generates x, y list of ints of length n_images

        Parameters
        ----------
        self: object
            A class object with parameters and attributes specified above

        Returns
        ----------
        _generated_x, _generated_y: list
            Lists of randomly generated ints of length n_images
        """
        z_lst = [np.shape(self.patient_data[indx][0])[2] for indx in patient_index]
#         print(z_lst)
        return([(self._gen_index(type = 'x'),
                 self._gen_index(type = 'y'),
                 self._gen_index(type = 'z', z_max = z_item)) for z_item in z_lst])


    def img_gen (self):
        """
        Function
        ----------
        Body of the function,

        Parameters
        ----------
        self: object
            A class object with parameters and attributes specified above

        Attributes
        ----------
        _patient_indx: list
            A list of randomly determined ints of length n_images
        _z_lst: list
            A list of the z-axis length of image groups in patient_data
        _x_lw, _x_hi, _y_lw, _y_hi, _z_lw, _z_hi: int
            Values corresponding to the index of patient images and labels

        Returns
        ----------
        generator: a generator object
            Returns a generator object which produces n_images specified croped images
                when called
        """
        if self.single_patient_batch == False and self.single_slice_batch == False:
            _patient_index = np.random.randint(low = 0, high = len(self.patient_data), size = self.n_sets)
        else:
#             print(len(self.patient_data), np.shape(self.patient_data))
            _patient_index = np.random.randint(low = 0, high = len(self.patient_data), size = self.n_sets)
            _patient_index[:] = _patient_index[0]
            
        self._x_max, self._y_max, _ = np.shape(self.patient_data[0][0])
        indx_lst = self._xyz_generator(patient_index = _patient_index)

        _img_batch = []
        _lbl_batch = []

        i_sl = -1
        for i, xyz_tpl in enumerate(indx_lst):
#             print('batch i: ', i)
            i_sl += 1
            x_lw, y_lw, z_lw = xyz_tpl
            x_hi, y_hi, z_hi = tuple(x + y for x,y in zip(xyz_tpl, self.dimensions))

            if i_sl == 0:
                z_lw_0 = z_lw
                z_hi_0 = z_hi
            if self.single_slice_batch == True:
                z_lw = z_lw_0
                z_hi = z_hi_0
#                 print(z_lw)
            
            _lbl_batch.append([])
            for i_ch in range(len(self.labels[0])):
                _lbl_batch[i].append(self.labels[_patient_index[i]][i_ch][x_lw : x_hi,  
                                                                          y_lw : y_hi,
                                                                          z_lw : z_hi])

            if self.normalized:
                _img_batch.append([])
                for i_ch in range(len(self.patient_data[0])):
                    img_grp = self.patient_data[_patient_index[i]][i_ch][x_lw : x_hi,
                                                                         y_lw : y_hi,
                                                                         z_lw : z_hi]
                    img_grp = (img_grp - np.min(img_grp))/(np.max(img_grp)- np.min(img_grp))
                    _img_batch[i].append(img_grp)
            else:
                _img_batch.append([])
                for i_ch in range(len(self.patient_data[0])):
                    _img_batch[i].append(self.patient_data[_patient_index[i]][i_ch][x_lw : x_hi,
                                                                                    y_lw : y_hi,
                                                                                    z_lw : z_hi])

        _img_batch = np.array(_img_batch, dtype = 'float32')
        _lbl_batch = np.array(_lbl_batch, dtype = 'float32')
#         print('pre:   ', np.shape(_img_batch), ' : ', np.shape(_lbl_batch))
#         print(self.Dim2_option)
        
        if self.Dim2_option:
#             print(len(_img_batch.shape))
            _img_batch = np.squeeze(_img_batch, axis=len(_img_batch.shape)-1)
            _lbl_batch = np.squeeze(_lbl_batch, axis=len(_lbl_batch.shape)-1)
#             print('dim2_option, post:   ', np.shape(_img_batch), ' : ', np.shape(_lbl_batch))
        
        if self.channels_last:
            _img_batch = np.rollaxis(_img_batch, 1, len(_img_batch.shape))
            _lbl_batch = np.rollaxis(_lbl_batch, 1, len(_lbl_batch.shape))
#             print('channels lase past:   ', np.shape(_img_batch), ' : ', np.shape(_lbl_batch))
        
        return(_img_batch, _lbl_batch)
        

    def return_generator (self, calls):
        """
        Function
        ----------
        Creates a generator of length calls

        Returns
        ----------
        Returns a <generator> of length self.epochs which randomly generates image
            groups of the specified dimensions
        """
        return(self.img_gen() for _ in range(calls))

#-------------------------------------------------------------------------------

class single_testing(object):
    def __init__(self, patient_data, labels, normalized = True,
                dimensions = (256,256,3), z_location = 1, n_sets = 1000,
                channels_last = True, **kwargs):
        self.patient_data = patient_data
        self.labels = labels
        self.normalized = normalized
        self.dimensions = dimensions
        self.z_location = z_location
        self.n_sets = n_sets
        self.channels_last = channels_last
        self._patient_index = np.random.randint(low = 0, high = len(self.patient_data), size = 1)
        self._x_max = np.shape(self.patient_data[0][0])[0]
        self._y_max = np.shape(self.patient_data[0][0])[1]
        self._x_indx = self._gen_val(type = 'x')
        self._y_indx = self._gen_val(type = 'y')
        self._z_indx = self._gen_val(type = 'z', z_max = np.shape(self.patient_data[self._patient_index][0])[2])
        self.lock = threading.Lock()
        self.img_single = None
        self.lbl_single = None


    def __iter__(self):
        return self

    def __next__(self):
        """
        Function
        ----------
        Returns values when the <generator> next is called

        Returns
        ----------
        Returns self.img_single, self.lbl_single
        """
        with self.lock:
            if self.img_single == None and self.lbl_single == None:
                self.single_image()
            else:
                return(self.img_single, self.lbl_single)

    def _gen_val(self, type, z_max = None):
        """
        Function
        ---------
        Creates a random integer between 0 and the axis max dimensions

        Parameters
        ----------
        type: str
            Specifies the type of limit that will be calculated
        z_max: int (Default = None)
            Specifies the maximum value for the generated z-axis index

        Returns
        ---------
        Returns the created integer
        """
        if type == 'x':
            if not(self._x_max-self.dimensions[0]):
                return(0)
            else:
                return(np.random.randint(low = 0, high = self._x_max-self.dimensions[0]))
        if type == 'y':
            if not(self._y_max-self.dimensions[1]):
                return(0)
            else:
                return(np.random.randint(low = 0, high = self._y_max-self.dimensions[1]))
        if type == 'z':
            return(np.random.randint(low = 0, high = z_max - self.dimensions[2]))

    def single_image(self):
        """
        Function
        ---------
        Produces a single image, repeated n_sets input_images

        Attributes
        ----------
        _img_batch: list
            A list of the batch of images
        _lbl_batch: list
            A list of the batch of labels
        _xyz_tpl: tuple
            A tuple of the x, y, z index values
        x_lw, x_hi, y_lw, y_hi, z_lw, z_hi: int
            Each represents the range of values for the given axis
        _img_grp: np.array
            A group of normalized images
        _img_single: np.array
            An array of _img_batch
        _lbl_single: np.array
            An array of _lbl_batch

        Returns
        ---------
        img_single
            A numpy array of _img_batch
        lbl_single
            A numpy array of _lbl_batch
        """

        _img_batch = []
        _lbl_batch = []
        _xyz_tpl = (self._x_indx, self._y_indx, self._z_indx)

        for _ in range(self.n_sets):
            x_lw, y_lw, z_lw = _xyz_tpl
            x_hi, y_hi, z_hi = tuple(x+y for x,y in zip(_xyz_tpl, self.dimensions))

            
            _lbl_batch.append([])
            for i_ch in range(len(self.labels[0])):
                _lbl_batch[i].append(self.labels[_patient_index[i]][i_ch][x_lw : x_hi,  
                                                                          y_lw : y_hi,
                                                                          z_lw : z_hi])

            if self.normalized:
                _img_batch.append([])
                for i_ch in range(len(self.patient_data[0])):
                    img_grp = self.patient_data[_patient_index[i]][i_ch][x_lw : x_hi,
                                                                         y_lw : y_hi,
                                                                         z_lw : z_hi]
                    img_grp = (img_grp - np.min(img_grp))/(np.max(img_grp)- np.min(img_grp))
                    _img_batch[i].append(img_grp)
            else:
                _img_batch.append([])
                for i_ch in range(len(self.patient_data[0])):
                    _img_batch[i].append(self.patient_data[_patient_index[i]][i_ch][x_lw : x_hi,
                                                                                    y_lw : y_hi,
                                                                                    z_lw : z_hi])
            
            
#             _lbl_batch.append(self.labels[self._patient_index, 
#                                           z_lw + self.z_location, 
#                                           x_lw:x_hi, y_lw:y_hi][0])

#             if self.normalized:
#                 _img_grp = self.patient_data[self._patient_index, z_lw:z_hi, 
#                                              x_lw:x_hi, y_lw:y_hi]
#                 _img_grp = (_img_grp-np.mean(_img_grp))/np.std(_img_grp)
#                 _img_batch.append(_img_grp)
#             else:
#                 _img_batch.append(self.patient_data[self._patient_index, z_lw:z_hi, 
#                                                     x_lw:x_hi, y_lw:y_hi])

        img_single = np.array(_img_batch, dtype='float32')
        lbl_single = np.array(_lbl_batch, dtype='float32')

        if self.channels_last:
            img_single = np.rollaxis(img_single, 1, len(img_single.shape))
            lbl_single = np.rollaxis(lbl_single, 1, len(lbl_single.shape))
        self.img_single = img_single
        self.lbl_single = lbl_single

        return(img_single, lbl_single)

    def single_img_generator(self, calls):
        """
        Function
        ---------
        Creates a generator object from the single_image funciton

        Returns
        ---------
        Returns a generator function for a single image
        """
        return(self.single_image() for _ in range(calls))

#-------------------------------------------------------------------------------

class limit_z(object):
    """
    Function
    ----------
    Limits the data surrounding the maximum slice value by the specified margins

    Parameters
    ----------
    labels: np.array
        An array of labels
    images: np.array
        An array of patient images
    margins: int
        A integer specifying the margins surrounding the maximum image value
    """

    def __init__ (self, labels, images, margins, window_level = True, wl_type = None, **kwargs):
        """
        Parameters
        ----------
        labels: np.array
            An array of the image labels
        images: np.array
            An array of the images
        window_level: bool (Default = True)
            A boolean specifying if the data should be windowed and leveled
        wl_type: str (Default = None)
            A string specifying which type of window and level preset should be used

        """
        self.labels = labels
        self.images = images
        self.margins = margins
        self.window_level = window_level
        self.wl_type = wl_type


    def window_level (self, input_images):
        """
        Function
        ---------
        To window or level an image to the standard values

        Parameters
        ----------
        input_images: np.array
            An np.array of the images that should be windowed and leveled

        Raises
        ----------
        KeyError
            Your specified wl_type is not within WL_Dict

        Returns
        ----------
        Returns the numpy array that is windowed and leveled as specified
        """
        # Values obtained from: https://radiopaedia.org/articles/ct-head-an-approach
        WL_Dict = {'Brain_Matter' : (80, 40),
                'Grey-White_Diff' : (40, 40),
                'Blood' : (130, 50),
                'Soft_Tissue' : (350, 40),
                'Bony_Review' : (2800, 600)}

        input_str = "Please specify the W/L type (bm, gw_d, bd, st, br):"
        options_dict = {
            bm_options : ['bm', 'brain matter', 'brain_matter', 'Brain Matter', 'Brain_Matter'],
            gw_options : ['gw_d', 'grey white diff', 'Grey-White Diff', 'Grey-White_Diff'],
            bd_options : ['bd', 'blood', 'Blood'],
            st_options : ['st', 'soft tissue', 'Soft Tissue', 'Soft_Tissue'],
            br_options : ['br', 'boney review', 'boney_review', 'Boney Review', 'Boney_Review'],
            custom : ['custom', 'Custom']}

        for key in options_dict:
            if self.wl_type in options_dict[key]:
                self.wl_type = options_dict[key][-1]
            else:
                raise KeyError('Your specified wl_type is within WL_Dict')

        if self.wl_type == None:
            self.wl_type = input(input_str)

        for key in list(options_dict.keys()):
            if self.wl_type in options_dict[key]:
                self.wl_type = options_dict[key][-1]
                break
        else:
            raise ValueError("The specifed WL type is not accepted")

        if self.wl_type == 'Custom':
            window = input("Specify the window value:")
            level = input("Specify the level value:")
        else:
            window, level = WL_Dict[self.wl_type]

        low_window_ct = level - window/2 + 1000
        high_window_ct = level + window/2 + 1000

        if self.exclued:
            input_images[input_images < low_window_ct] = 0
            input_images[input_images > high_window_ct] = 0
        else:
            input_images[input_images < low_window_ct] = 0
            input_images[input_images > high_window_ct] = input_images.max()
        return(wl_image)


    def _max_z (self, reduced_arr):
        """
        Function
        ----------
        Calculates the sum of each z-slice in the provided array

        Parameters
        ----------
        reduced_array: np.array
            The array to be reduced

        Returns
        ----------
        reduced_array: np.array
            Returns the provided array with dimesnions requested
        """
        if len(reduced_arr.shape) <= 1:
            return(reduced_arr.argmax())
        else:
            return(self._max_z(reduced_arr = reduced_arr.sum(-1)))


    def _data_limiter (self):
        """
        Function
        ----------
        Produces limited arrays of images surrounding the maximum label slice with
            margins specified

        Parameters
        ----------
        labels: np.array
            An array of labels in the form (patient, z, x, y, channel)
        images: np.array
            An array of images in the form (patient, z, x, y, channel)

        Attributes
        ----------
        _max_labels: list
            A list of the maximum label sums in for a given z-slice

        Returns
        ----------
        crop_label: np.array
            A cropped label array centered around the maximum summed label with
                margins as specified
        crop_images: np.array
            A cropped image array centered around the maximum summed label with
                margins as specified
        """
        _max_labels = [self._max_z(reduced_arr = label) for label in self.labels]
        crop_label = []
        crop_images = []

        for index, max_lbl in enumerate(_max_labels):
            low = max(max_lbl - self.margins, 0)
            high = min(max_lbl + self.margins, self.labels[index].shape[0])
            crop_label.append(self.labels[index][low:high, :, :])
            crop_images.append(self.images[index][low:high, :, :])

        crop_label = np.array(crop_label)
        crop_images = np.array(crop_images)

        return(crop_images, crop_label.astype(int))





