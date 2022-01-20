# Name: Odysseas Papakyriakou

# only to be used within the linux subsystem
import nibabel
import nighres
from pathlib import Path
import datetime


class Delineations:    
    def __init__(self, subs, structure, side, rater, delineations_dir, out_path, tmp_path, today):
        """Constructor. Always specify these parameters when creating a Delineations object"""
        
        self.subs = subs
        self.structure = structure
        self.side = side
        self.rater = rater
        self.delineations_dir = delineations_dir
        self.out_path = out_path
        self.tmp_path = tmp_path
        self.today = today  

    def getSingleVolume(self, mask, stats_file):
        """Calculates the volume for one mask and outputs it in a csv file.
        
        Parameters
        ----------
        mask : pathlib.Path
               the directory of the mask
        stats_file : pathlib.Path
                     the directory and name of the output csv file        
        """
        
        img1 = nighres.io.load_volume(str(mask))
        data1 = img1.get_data()
        hdr = img1.get_header()
        aff = img1.get_affine()
        imgres = [x.item() for x in hdr.get_zooms()]
        imgdim = data1.shape
        bin1 = nibabel.Nifti1Image(data1>0, aff, hdr)
        
        # save the binarized images with proper names etc
        bin1_file = tmp_path / Path(str(Path(Path(mask).stem).stem)+'_calc-bin.nii.gz')
        nighres.io.save_volume(str(bin1_file), bin1)
        
        # calculate the volume
        nighres.statistics.segmentation_statistics(str(bin1_file),
                                                   statistics=['Volume'],
                                                   output_csv=str(stats_file))
        
    def getVolumes(self, masks, out_path, tmp_path):
        """Calculates the volume for all masks using the getSingleVolume method
        
        Parameters
        ----------
        masks : list 
                a list of pathlib.Path objects showing the directory of each mask
        out_path : pathlib.Path
                   the output location of the csv file
        tmp_path : pathlib.Path
                   the location of the temporary files
        """
        
        out_path.mkdir(exist_ok=True)
        tmp_path.mkdir(exist_ok=True)
        stats_file = out_path / Path('statistics-single_mask-'+structure+'_hem-'+side+'_date-'+today+'.csv')
        for mask in masks:
            Delineations.getSingleVolume(self, mask, stats_file)
            
    def getMasks(self):
        """Returns a list of pathlib.Path objects to the location of the masks"""
        
        final_masks = []
        for sub in subs:
            sub = str(sub).zfill(3)
            # I kept the same directories as the ones in the server
            mask_path =  delineations_dir/('sub-'+sub)/'ses-1'/'derivatives'/structure/'manual_masks'
            
            # find existing masks
            print('looking for masks: '+'sub-'+sub+'*_mask-'+structure+'_hem-'+side+'*.nii.gz')
            masks = list(mask_path.glob('sub-'+sub+'*_mask-'+structure+'_hem-'+side+'*.nii.gz'))
            print(str(len(masks))+' masks found')
            
            # check if there is at least one mask:
            if len(masks)>=1:
                mask = list(mask_path.glob('sub-'+sub+'*_mask-'+structure+'_hem-'+side+'_rat-'+rater+'.nii.gz'))
                if len(mask)==1:
                    final_masks.append(mask[0])
        
        return final_masks
         

if __name__ == "__main__":
    """Specify the parameters here. This code will not be executed if the modules are imported later"""
    
    subs = [16,23,35,44,54,55,60,75,76,82,84,
            111,112,113,114,115,116,117,118,119,120,121]
    structure = "mgn"
    side = str(input("Type 'l' for the left hemisphere OR 'r' for the right hemisphere: "))
    rater = "opx"
    delineations_dir = Path("mri_delineations")
    out_path = Path("odysseas_data")
    tmp_path = Path("tmp_odysseas_data")
    today = str(datetime.date.today())
    
    # create a Delineations object with the specified parameters
    delineations_object = Delineations(subs, structure, side, rater, delineations_dir, out_path, tmp_path, today)
    # get the masks
    masks = delineations_object.getMasks()
    # calculate volume for all masks
    delineations_object.getVolumes(masks, out_path, tmp_path)
