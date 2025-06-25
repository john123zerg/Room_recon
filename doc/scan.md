# 📱 Scanning to Reconstruction Guide with NerfCapture

## Requirements

- iPhone Pro 14 or later (LiDAR required)
- macOS with Xcode installed

---

## ✅ Step-by-Step Installation

### 1. Install Xcode on Mac

Download Xcode from the official site:  
👉 https://developer.apple.com/xcode

<img src="https://github.com/user-attachments/assets/41687e22-a0ef-4199-b67d-1a83f004f256" width="200"/>

---

### 2. Clone the NerfCapture Repository


```bash
git clone https://github.com/Zhangyangrui916/NeRFCapture.git
```


```bash
After cloning, connect your iPhone to your Mac via USB and open the project in Xcode.

```
### 3. Set Up the Xcode Project

- Open `NeRFCapture.xcodeproj` in Xcode.
- Install any missing components for iOS development if prompted.
- Connect your iPhone and ensure it appears as a run target.

<img src="https://github.com/user-attachments/assets/8c3d42b2-8165-4139-acfb-54addd469bc4" width="400"/>

- Press **Build**.

<img src="https://github.com/user-attachments/assets/d7672240-ff46-46b4-976e-414882b84f77" width="400"/>

---

### 4. Enable Developer Mode on iPhone

- On your iPhone, go to:  
  `Settings → VPN & Device Management`

<img src="https://github.com/user-attachments/assets/afd6a57f-8124-41c4-bc76-b96533e0982b" width="400"/>

- Approve the developer certificate and enable **Developer Mode**  
  (you might find it under **Privacy & Security** on newer iOS versions)

<img src="https://github.com/user-attachments/assets/c358f081-7e5d-4001-8a7f-9ae73ce3ca55" width="374"/> <img src="https://github.com/user-attachments/assets/8848ec81-2a75-4cab-8ea3-15422923441a" width="182"/>

---

### 5. Launch the NeRFCapture App

- Once the app is installed, open **NeRFCapture** on your iPhone.

<img src="https://github.com/user-attachments/assets/3cef57c2-09e0-49a5-a99e-eb749a4f9392" width="400"/> <img src="https://github.com/user-attachments/assets/4a8eacef-1678-44d2-9548-2475477ca51a" width="100"/>

---

### 6. Scanning Workflow

- Tap **Reset** to define the origin frame.

<img src="https://github.com/user-attachments/assets/4b688c14-6ef5-497a-9c2f-8b9637b6aa09" width="400"/>

- Tap **Start** to begin capturing frames.

<img src="https://github.com/user-attachments/assets/18934dd8-b76b-4760-b44e-7ce0106fe742" width="400"/>

- Tap **End** when done.

---


### 7. Upload files

Go to 'Files' and you will see a NeRFCapture folder.
Sometimes the zip file is corrupted, so it is best to compress the original image data.
Then, you can share the data to test the pipeline.

---

### ⚠️ Notes & Tips

- 📸 Try to hold the phone steady while scanning — motion blur can degrade quality. (e.g., pause for 1.5 ~ 2 seconds after each frame)
  - Also, try to overlap at least 30% of the previous frame for better quality.
- 🗜️ If the ZIP export is corrupted, re-compress the original folder manually before processing.
