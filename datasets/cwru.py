import os
import re
import scipy.io
import numpy as np
from datasets.base_dataset import BaseDataset
from torchvision.datasets import ImageFolder

class CWRU(BaseDataset):    

    def list_of_bearings(self):
        """ 
        Returns: 
            A list of tuples containing filenames (for naming downloaded files) and URL suffixes 
            for downloading vibration data.
        """
        all_bearings = [
        ("97", "97.mat"),   ("98", "98.mat"),   ("99", "99.mat"),   ("100", "100.mat"), ("105", "105.mat"), 
        ("106", "106.mat"), ("107", "107.mat"), ("108", "108.mat"), ("109", "109.mat"), ("110", "110.mat"), 
        ("111", "111.mat"), ("112", "112.mat"), ("118", "118.mat"), ("119", "119.mat"), ("120", "120.mat"), 
        ("121", "121.mat"), ("122", "122.mat"), ("123", "123.mat"), ("124", "124.mat"), ("125", "125.mat"), 
        ("130", "130.mat"), ("131", "131.mat"), ("132", "132.mat"), ("133", "133.mat"), ("135", "135.mat"), 
        ("136", "136.mat"), ("137", "137.mat"), ("138", "138.mat"), ("144", "144.mat"), ("145", "145.mat"), 
        ("146", "146.mat"), ("147", "147.mat"), ("148", "148.mat"), ("149", "149.mat"), ("150", "150.mat"), 
        ("151", "151.mat"), ("156", "156.mat"), ("158", "158.mat"), ("159", "159.mat"), ("160", "160.mat"), 
        ("161", "161.mat"), ("162", "162.mat"), ("163", "163.mat"), ("164", "164.mat"), ("169", "169.mat"), 
        ("170", "170.mat"), ("171", "171.mat"), ("172", "172.mat"), ("174", "174.mat"), ("175", "175.mat"), 
        ("176", "176.mat"), ("177", "177.mat"), ("185", "185.mat"), ("186", "186.mat"), ("187", "187.mat"), 
        ("188", "188.mat"), ("189", "189.mat"), ("190", "190.mat"), ("191", "191.mat"), ("192", "192.mat"), 
        ("197", "197.mat"), ("198", "198.mat"), ("199", "199.mat"), ("200", "200.mat"), ("201", "201.mat"), 
        ("202", "202.mat"), ("203", "203.mat"), ("204", "204.mat"), ("209", "209.mat"), ("210", "210.mat"), 
        ("211", "211.mat"), ("212", "212.mat"), ("213", "213.mat"), ("214", "214.mat"), ("215", "215.mat"), 
        ("217", "217.mat"), ("222", "222.mat"), ("223", "223.mat"), ("224", "224.mat"), ("225", "225.mat"), 
        ("226", "226.mat"), ("227", "227.mat"), ("228", "228.mat"), ("229", "229.mat"), ("234", "234.mat"), 
        ("235", "235.mat"), ("236", "236.mat"), ("237", "237.mat"), ("238", "238.mat"), ("239", "239.mat"), 
        ("240", "240.mat"), ("241", "241.mat"), ("246", "246.mat"), ("247", "247.mat"), ("248", "248.mat"), 
        ("249", "249.mat"), ("250", "250.mat"), ("251", "251.mat"), ("252", "252.mat"), ("253", "253.mat"), 
        ("258", "258.mat"), ("259", "259.mat"), ("260", "260.mat"), ("261", "261.mat"), ("262", "262.mat"), 
        ("263", "263.mat"), ("264", "264.mat"), ("265", "265.mat"), ("270", "270.mat"), ("271", "271.mat"), 
        ("272", "272.mat"), ("273", "273.mat"), ("274", "274.mat"), ("275", "275.mat"), ("276", "276.mat"), 
        ("277", "277.mat"), ("278", "278.mat"), ("279", "279.mat"), ("280", "280.mat"), ("281", "281.mat"), 
        ("282", "282.mat"), ("283", "283.mat"), ("284", "284.mat"), ("285", "285.mat"), ("286", "286.mat"), 
        ("287", "287.mat"), ("288", "288.mat"), ("289", "289.mat"), ("290", "290.mat"), ("291", "291.mat"), 
        ("292", "292.mat"), ("293", "293.mat"), ("294", "294.mat"), ("295", "295.mat"), ("296", "296.mat"), 
        ("297", "297.mat"), ("298", "298.mat"), ("299", "299.mat"), ("300", "300.mat"), ("301", "301.mat"), 
        ("302", "302.mat"), ("305", "305.mat"), ("306", "306.mat"), ("307", "307.mat"), ("309", "309.mat"), 
        ("310", "310.mat"), ("311", "311.mat"), ("312", "312.mat"), ("313", "313.mat"), ("315", "315.mat"), 
        ("316", "316.mat"), ("317", "317.mat"), ("318", "318.mat"), ("3001", "3001.mat"), ("3002", "3002.mat"), 
        ("3003", "3003.mat"), ("3004", "3004.mat"), ("3005", "3005.mat"), ("3006", "3006.mat"), ("3007", "3007.mat"), 
        ("3008", "3008.mat")
        ]
        
        # If not using domain split, just return all files.
        if not self.use_domain_split:
            return all_bearings

        # Defensive check for empty/null train_domains or test_domain
        if not self.train_domains or not self.test_domain:
            return all_bearings
        
        if self.use_domain_split:
            selected_files = set()
            for domain in self.train_domains + [self.test_domain]:
                semantic_names = self.domain_mapping.get(domain, [])
                for name in semantic_names:
                    mapped = self.semantic_to_file.get(name)
                    if mapped:
                        selected_files.add(mapped)
                    else:
                        print(f"[WARNING] No mapping found for {name}")
            return [b for b in all_bearings if b[0] in selected_files]

        return all_bearings

    def __init__(self, use_domain_split=True, train_domains=None, test_domain=None):

        super().__init__(rawfilesdir = "data/raw/cwru",
                         url = "https://engineering.case.edu/sites/default/files/")
        
        self.use_domain_split = use_domain_split
        self.train_domains = train_domains if train_domains else []
        self.test_domain = test_domain if test_domain else ""

        # Define domain mapping (based on your Table 2 CWRU 48k DE split)
        self.domain_mapping = {
            "1": ["Normal_0", "IR007_0", "OR007@6_0", "B007_0"],
            "2": ["Normal_1", "IR007_1", "OR007@6_1", "B007_1"],
            "3": ["Normal_2", "IR007_2", "OR007@6_2", "B007_2"],
            "4": ["Normal_3", "IR007_3", "OR014@6_3", "B007_3"],
            "5": ["Normal_0", "IR014_0", "OR014@6_0", "B014_0"],
            "6": ["Normal_1", "IR014_1", "OR014@6_1", "B014_1"],
            "7": ["Normal_2", "IR014_2", "OR014@6_2", "B014_2"],
            "8": ["Normal_3", "IR014_3", "OR014@6_3", "B014_3"],
            "9": ["Normal_0", "IR021_0", "OR021@6_0", "B021_0"],
            "10": ["Normal_1", "IR021_1", "OR021@6_1", "B021_1"],
            "11": ["Normal_2", "IR021_2", "OR021@6_2", "B021_2"],
            "12": ["Normal_3", "IR021_3", "OR021@6_3", "B021_3"]
        }
        self.semantic_to_file = {
            # Domain 1
            "Normal_0": "97",
            "IR007_0": "105",
            "OR007@6_0": "118",
            "B007_0": "130",

            # Domain 2
            "Normal_1": "98",
            "IR007_1": "106",
            "OR007@6_1": "119",
            "B007_1": "131",

            # Domain 3
            "Normal_2": "99",
            "IR007_2": "107",
            "OR007@6_2": "120",
            "B007_2": "132",

            # Domain 4
            "Normal_3": "100",
            "IR007_3": "108",
            "OR014@6_3": "121",
            "B007_3": "133",

            # Domain 5
            "IR014_0": "109",
            "OR014@6_0": "122",
            "B014_0": "135",

            # Domain 6
            "IR014_1": "110",
            "OR014@6_1": "123",
            "B014_1": "136",

            # Domain 7
            "IR014_2": "111",
            "OR014@6_2": "124",
            "B014_2": "137",

            # Domain 8
            "IR014_3": "112",
            "OR014@6_3": "125",
            "B014_3": "138",

            # Domain 9
            "IR021_0": "144",
            "OR021@6_0": "145",
            "B021_0": "146",

            # Domain 10
            "IR021_1": "147",
            "OR021@6_1": "148",
            "B021_1": "149",

            # Domain 11
            "IR021_2": "150",
            "OR021@6_2": "151",
            "B021_2": "156",

            # Domain 12
            "IR021_3": "158",
            "OR021@6_3": "159",
            "B021_3": "160",
        }

        if self.use_domain_split:
            print(f">> CWRU Initialized | Train Domains: {self.train_domains} | Test Domain: {self.test_domain}")
        else:
            print(f">> CWRU Initialized in **Non-Split Mode** (Entire Dataset Used)")

    def get_data_by_domain(self, domain):
        """
        Retrieves data for a specific domain.
        """
        if self.use_domain_split:
            if domain in self.train_domains:
                print(f"Loading ***Training/Validation*** data for domain {domain}")
            elif domain == self.test_domain:
                print(f"Loading ***Test*** data for domain {domain}")
            else:
                raise ValueError(f"Domain {domain} is not configured for this dataset.")
        else:
            print(f"Loading **Full CWRU Dataset (No domain split)**")

        return self.load_data_from_directory(domain)
    
    def get_domain_folder(self, is_test=False):
        if self.use_domain_split:
            if is_test:
                return f"test_domain_{self.test_domain}"
            return f"train_domains_{'_'.join(self.train_domains)}"
        return ""

    def load_data_from_directory(self, domain):
        """
        Load spectrograms from the corresponding domain directory.
        """
        domain_path = os.path.join("data/spectrograms/cwru", domain)
        if not os.path.exists(domain_path):
            raise FileNotFoundError(f"Missing spectrogram directory: {domain_path}")

        print(f"Loading spectrograms from {domain_path}")
        return ImageFolder(domain_path)

    def _extract_data(self, filepath):
        """ Extracts data from a .mat file for bearing fault analysis.
        Args:
            filepath (str): The path to the .mat file.
        Return:
            tuple: A tuple containing the extracted data and its label.
        """
        matlab_file = scipy.io.loadmat(filepath)
        keys = re.findall(r'X\d{3}_[A-Z]{2}_time', str(matlab_file.keys()))
        map_position = {
            '6203': 'FE',
            '6205': 'DE' }
        basename = os.path.basename(filepath).split('.')[0]
        annot_info = list(filter(lambda x: x["filename"]==basename, self.annotation_file))[0]
        
        if annot_info is None:
            raise ValueError(f"No annotation found for {basename}")
        
        label = annot_info["label"]
        bearing_type = annot_info["bearing_type"]
        bearing_position = ['DE'] if label == 'N' else [map_position[bearing_type]]
        for key in keys:
            if key[-7:-5] in bearing_position:
                data_squeezed = np.squeeze(matlab_file[key])  # removes the dimension corresponding to 
                                                              # the number of channels, as only a single channel is being used.
                if self.acquisition_maxsize:
                    return data_squeezed[:self.acquisition_maxsize], label
                else:
                    return data_squeezed, label

    def __str__(self):
        return "CWRU"
    