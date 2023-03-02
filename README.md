# Note

## Colab + Google Drive

### `gdown`

- [https://github.com/wkentaro/gdown](https://github.com/wkentaro/gdown)

- Update `gdown` (in case having problem)

```bash
!pip install --upgrade --no-cache-dir gdown > /dev/null
```

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
