# Note

## Convert Notebook to Markdown doc

```bash
cd .\01-pytorch-basics
jupyter nbconvert torch.ipynb --to markdown --output README.md
```

- **Issues**:  `nbconvert` failing to convert notebook to markdown:

```bash
pip install "nbconvert<6"
```

## logless installation

```py
%%capture
!pip install x
```
Or
```py
!pip install x > /dev/null
```

### vim in colab:

Remap `jj` to <ESC> in Google Colab Vim:
- press `escape` or `CTRL+[` to switch normal mode then press : to switch to command mode then type: `:imap jk <Esc>`

### `gdown`

- [https://github.com/wkentaro/gdown](https://github.com/wkentaro/gdown)

- Update `gdown` (in case having problem)

```bash
!pip install --upgrade --no-cache-dir gdown > /dev/null
```
From Command Line:

```python
# download with fuzzy extraction of a file ID
!gdown --fuzzy 'https://drive.google.com/file/d/1Sht13qgc-uJ41LNtYjDiY_veoA5ApUEv/view?usp=share_link'
# a folder
!gdown https://drive.google.com/drive/folders/1bsaNHhwhfD8t1d2dlSItS_32Phjlv4bu?usp=share_link -O /content/results --folder
```

From Python:
```python
import gdown

# a file
url = "https://drive.google.com/uc?id=1l_5RK28JRL19wpT22B-DY9We3TVXnnQQ"
output = "fcn8s_from_caffe.npz"
gdown.download(url, output, quiet=False)

# same as the above, but with the file ID
id = "0B9P1L--7Wd2vNm9zMTJWOGxobkU"
gdown.download(id=id, output=output, quiet=False)

# same as the above, and you can copy-and-paste a URL from Google Drive with fuzzy=True
url = "https://drive.google.com/file/d/0B9P1L--7Wd2vNm9zMTJWOGxobkU/view?usp=sharing"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)

# a folder
url = "https://drive.google.com/drive/folders/15uNXeRBIhVvZJIhL4yTw4IsStMhUaaxl"
gdown.download_folder(url, quiet=True, use_cookies=False)

# same as the above, but with the folder ID
id = "15uNXeRBIhVvZJIhL4yTw4IsStMhUaaxl"
gdown.download_folder(id=id, quiet=True, use_cookies=False)
```

### Misc.

-  Delete folder with it's content

```python
import shutil
shutil.rmtree('/content/data/info') #deletes a directory and all its contents.
```

- unzip

```bash
!unzip "/content/info.zip" -d "/content/data" > /dev/null
```

- centering image in markdown

```html
<div align="center">
<img src="img/file_name" alt="file_name" width="width00px">
</div>
```

-  Generating `README.md`

```bash
cd ~
jupyter nbconvert --to markdown file_name --output README.md
```

## Mounting Gdrive in Colab

- Give permission

```python
from google.colab import drive
drive.mount('/content/drive')
```

- Usage

```python
output_path = f"/content/drive/MyDrive/result.csv"
results_df.to_csv(output_path, index=False)
```

## Tensorflow Helper Functions

Download:

```bash
!wget https://raw.githubusercontent.com/dev-SR/Deep-Learning/main/tf_helper_functions.py
```

Import:

```python
from tf_helper_functions import (
    delete_dir,
    split_dataset_train_test,
    walk_through_dir,
    create_subset_dataset,
    view_random_image,
    augment_random_image,
    plot_loss_curves_mplt,
    plot_loss_curves_plotly,
    create_tensorboard_callback,
    create_feature_extractor_model,
    compare_histories_plotly,
    compare_histories_mplt,
    get_callbacks
)
```
