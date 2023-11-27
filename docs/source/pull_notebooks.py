import requests

tutorials = [
    "https://raw.github.com/Ciela-Institute/caustics-tutorials/main/tutorials/BasicIntroduction.ipynb",
    "https://raw.github.com/Ciela-Institute/caustics-tutorials/main/tutorials/LensZoo.ipynb",
    "https://raw.github.com/Ciela-Institute/caustics-tutorials/main/tutorials/VisualizeCaustics.ipynb",
    "https://raw.github.com/Ciela-Institute/caustics-tutorials/main/tutorials/MultiplaneDemo.ipynb",
    "https://raw.github.com/Ciela-Institute/caustics-tutorials/main/tutorials/InvertLensEquation.ipynb",
]
for url in tutorials:
    try:
        R = requests.get(url)
        with open(url[url.rfind("/") + 1 :], "w") as f:
            f.write(R.text)
    except:
        print(
            f"WARNING: couldn't find tutorial: {url[url.rfind('/')+1:]} check internet conection"
        )
