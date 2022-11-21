from torch.utils.data import Dataset



class Lock3Dface(Dataset):
    def __init__(self, datafile, num2id,
                 add_noise=False,
                 using_normalmap=True,
                 is_training=True,
                 transform=None) -> None:
        super().__init__()
        self.img_paths = np.loadtxt(datafile, dtype=str).tolist()
        self.num2id = num2id
        self.add_noise = add_noise
        self.using_normalmap = using_normalmap
        self.is_training = is_training
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = cv.imread(img_path, flags=cv.IMREAD_UNCHANGED)

        if self.add_noise and self.is_training and random.random() < 0.4:
            img[:, :, 0] = add_noise_Guass(img[:, :, 0], mean=0, var=2e-5)

        if not self.using_normalmap:
            """depth"""
            img = img[:, :, 0]
        dir_name = img_path.split(sep='/')[-2]  # 不同系统可能有差异，注意
        num = int(dir_name[:3])
        label = self.num2id[num]
        # img = torch.tensor(img, dtype=torch.float32)
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        # img = torch.permute(img, (2, 0, 1))
        label = torch.tensor(label)
        subsets = {'NU':0, 'FE': 1, 'PS':2, 'OC':3, 'TM':4}
        basic_subset = subsets[dir_name.split('_')[-2]]  # //NU FE PS OC //TM
        basic_subset = torch.tensor(basic_subset)
        TM_subset = False
        if num in list(self.num2id.keys())[-169:]:
            TM_subset = True
        TM_subset = torch.tensor(TM_subset)

        return img, label, basic_subset, TM_subset

    def __len__(self):
        return len(self.img_paths)