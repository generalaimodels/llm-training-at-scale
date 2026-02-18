# Media Rendering – Full Capability Test
**Markdown + HTML | Real URLs | Copy–Paste Ready**

---

## 1. Images (Markdown)

![Big Buck Bunny](http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/images/BigBuckBunny.jpg)
![Elephants Dream](http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/images/ElephantsDream.jpg)
![Sintel](http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/images/Sintel.jpg)

---

## 2. Images (HTML with Size & Alignment)

<img src="http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/images/ForBiggerBlazes.jpg" width="300" />

<p align="center">
  <img src="http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/images/ForBiggerEscapes.jpg" width="400" />
</p>

---

## 3. Image Grid (HTML Table)

<table>
  <tr>
    <td><img src="http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/images/ForBiggerFun.jpg" width="220"/></td>
    <td><img src="http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/images/ForBiggerJoyrides.jpg" width="220"/></td>
  </tr>
  <tr>
    <td><img src="http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/images/ForBiggerMeltdowns.jpg" width="220"/></td>
    <td><img src="http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/images/TearsOfSteel.jpg" width="220"/></td>
  </tr>
</table>

---

## 4. Audio – Inline Playback (HTML)

### MP3
<audio controls>
  <source src="https://github.com/SergLam/Audio-Sample-files/raw/master/sample.mp3" type="audio/mpeg">
</audio>

### Multiple Formats (Fallback)
<audio controls>
  <source src="https://github.com/SergLam/Audio-Sample-files/raw/master/sample.m4a" type="audio/mp4">
  <source src="https://github.com/SergLam/Audio-Sample-files/raw/master/sample.ogg" type="audio/ogg">
  <source src="https://github.com/SergLam/Audio-Sample-files/raw/master/sample.wav" type="audio/wav">
</audio>

---

## 5. Audiogram / LRC Files

- [sample.lrc](https://github.com/SergLam/Audio-Sample-files/raw/master/sample.lrc)
- [sample1.lrc](https://github.com/SergLam/Audio-Sample-files/raw/master/sample1.lrc)
- [sample3.lrc](https://github.com/SergLam/Audio-Sample-files/raw/master/sample3.lrc)

---

## 6. Video – Inline Player with Poster

<video
  controls
  width="640"
  poster="http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/images/BigBuckBunny.jpg">
  <source src="http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4" type="video/mp4">
</video>

---

## 7. Video – Autoplay / Loop (Muted)

<video width="360" muted autoplay loop playsinline>
  <source src="http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerFun.mp4" type="video/mp4">
</video>

---

## 8. Clickable Image → Video (Pure Markdown)

[![Elephants Dream](http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/images/ElephantsDream.jpg)]
(http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4)

[![Tears of Steel](http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/images/TearsOfSteel.jpg)]
(http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/TearsOfSteel.mp4)

---

## 9. Mixed Media Card Layout

<table>
  <tr>
    <td width="50%">
      <img src="http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/images/SubaruOutbackOnStreetAndDirt.jpg" width="100%"/>
    </td>
    <td width="50%">
      <strong>Subaru Outback</strong><br/>
      <video controls width="100%">
        <source src="http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/SubaruOutbackOnStreetAndDirt.mp4" type="video/mp4">
      </video>
    </td>
  </tr>
</table>

---

## 10. Direct Media Links (Validation)

- [Big Buck Bunny – MP4](http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4)
- [Sintel – MP4](http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/Sintel.mp4)
- [Sample MP3](https://github.com/SergLam/Audio-Sample-files/raw/master/sample.mp3)

---

## Notes

- Uses **only real, publicly accessible URLs**
- Markdown + HTML hybrid
- Compatible with **GitHub Flavored Markdown**
- Tables used for layout/grid testing
- Autoplay depends on host/browser policy

## Local Media Note

Local large media binaries are intentionally excluded from the GitHub Pages build artifact to keep repository and deployment size controlled.  
Use the remote media sections above for production-valid rendering tests.
