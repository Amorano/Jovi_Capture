<div align="center">

<picture>
  <source srcset="https://raw.githubusercontent.com/Amorano/Jovimetrix-examples/refs/heads/master/res/logo-jovi_capture.png">
  <img alt="Capture Webcamera and URL media streams as ComfyUI images" width="256" height="256">
</picture>

</div>

<div align="center">

<a href="https://github.com/comfyanonymous/ComfyUI">COMFYUI</a> Nodes for capturing web camera and URL media streams as ComfyUI images

</div>

<div align="center">

![KNIVES!](https://badgen.net/github/open-issues/Amorano/Jovi_Capture)
![FORKS!](https://badgen.net/github/forks/Amorano/Jovi_Capture)

</div>

<!---------------------------------------------------------------------------->

# SPONSORSHIP

Please consider sponsoring me if you enjoy the results of my work, code or documentation or otherwise. A good way to keep code development open and free is through sponsorship.

<div align="center">

&nbsp;|&nbsp;|&nbsp;|&nbsp;
-|-|-|-
[![BE A GITHUB SPONSOR ❤️](https://img.shields.io/badge/sponsor-30363D?style=for-the-badge&logo=GitHub-Sponsors&logoColor=#EA4AAA)](https://github.com/sponsors/Amorano) | [![DIRECTLY SUPPORT ME VIA PAYPAL](https://img.shields.io/badge/PayPal-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://www.paypal.com/paypalme/onarom) | [![PATREON SUPPORTER](https://img.shields.io/badge/Patreon-F96854?style=for-the-badge&logo=patreon&logoColor=white)](https://www.patreon.com/joviex) | [![SUPPORT ME ON KO-FI!](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/alexandermorano)
</div>

## HIGHLIGHTS

<div align="center">
<img src="https://github.com/user-attachments/assets/cc7a6483-401b-4049-9645-6f57a940fb70" alt="Jovi Streaming Node Family" width="768"/>
</div>

* `WINDOW` capture desktop windows or window regions
* `CAMERA` webcam streaming and capture
* `REMOTE` capture remote URL media streams
* `MONITOR` snapshot local desktop monitor screen(s) or regions

## UPDATES

**2025/05/31** @1.1.3:
* removed extraneous debug information

**2025/05/31** @1.1.2:
* updated to comfy_cozy 0.0.32
* fixed bug in size capture when using non-zero offsets

**2025/05/16** @1.1.1:
* updated to comfy_cozy 0.0.25

**2025/04/30** @1.1.0:
* widget definitions formatted for clarity
* align names to Lexicon in comfy_cozy

**2025/04/30** @1.0.16:
* VEC2INT converted to VEC2
* better timeout default

**2025/04/30** @1.0.15:
* cleaned up old JS supports
* new comfy-cozy version
* widget_vector unified to match Jovimetrix

**2025/04/19** @1.0.14:
* removed old vector conversions waiting for new frontend mechanism

**2025/04/12** @1.0.13:
* reduced dependancy on torch constructs

**2025/04/12** @1.0.12:
* updated requirements for numpy to be < 2.0.0

**2025/03/28** @1.0.11:
* updated requirements for numpy to only be >=1.26.4

**2025/03/18** @1.0.10:
* dunno how to work around comfyui not doing type conversion

**2025/03/18** @1.0.5:
* vectors now convert in-line

**2025/03/08** @1.0.3:
* removed security scanner failures for in-line http links

**2025/03/02** @1.0.2:
* adjusted js import paths

**2025/03/01** @1.0.1:
* pywinctl for monitor capture

**2025/03/01** @1.0.0:
* intial breakout from Jovimetrix

# INSTALLATION

## COMFYUI MANAGER

If you have [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager) installed, simply search for Jovi_Capture and install from the manager's database.

## MANUAL INSTALL
Clone the repository into your ComfyUI custom_nodes directory. You can clone the repository with the command:
```
git clone https://github.com/Amorano/Jovi_Capture.git
```
You can then install the requirements by using the command:
```
.\python_embed\python.exe -s -m pip install -r requirements.txt
```
If you are using a <code>virtual environment</code> (<code><i>venv</i></code>), make sure it is activated before installation. Then install the requirements with the command:
```
pip install -r requirements.txt
```
# WHERE TO FIND ME

You can find me on [![DISCORD](https://dcbadge.vercel.app/api/server/62TJaZ3Z5r?style=flat-square)](https://discord.gg/62TJaZ3Z5r).
