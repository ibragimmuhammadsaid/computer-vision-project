## Dataset folder structure (IMPORTANT)

After downloading the dataset, **extract the main archive first**.  
In many cases, inside the extracted dataset you will still find additional archives inside subfolders — **you must extract each of those sub-archives separately** until you see real image files (e.g., `*.jpg`).

### Expected structure (placeholders)

After everything is extracted correctly, you should have folders like this:

data/
frames/
frames0X/
1CMX_X_R_#XXX/
*.jpg
1CMX_X_R_#XXX/
*.jpg
...
frames0X/
...


- `frames0X/` means folders like `frames01`, `frames02`, etc.
- `1CMX_X_R_#XXX/` means folders like `1CM1_1_R_#217`, `1CM2_3_R_#105`, etc.
- Inside each `1CM...` folder you should see **JPEG frame images** (`*.jpg`).

✅ **Correct extraction check:**  
If you open the folders and see actual `*.jpg` files inside, the dataset is extracted correctly.  
❌ If you still see `*.zip/.rar/.7z` files inside these folders, extract them too.
