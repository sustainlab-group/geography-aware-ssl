import urllib.request, urllib.parse, urllib.error
import json

FLICKR_API_KEY = "<INSERT API KEY>"  # obtained from Flickr after getting a license

FLICKR_API_BASE_URL = (
        "https://api.flickr.com/services/rest/?api_key=%s&format=json&nojsoncallback=1"
        % FLICKR_API_KEY
)


def get_photo_location(photo_id):
    url = (
            FLICKR_API_BASE_URL
            + "&method=flickr.photos.geo.getLocation&photo_id=%d" % photo_id
    )
    result = None
    try:
        with urllib.request.urlopen(url) as r:
            rsp = json.loads(r.read().decode("utf-8"))
            if "stat" in rsp and rsp["stat"] == "ok":
                result = {"id": photo_id}
                info = rsp["photo"]["location"]
                for k in ["latitude", "longitude"]:
                    result[k[:3]] = info.get(k, -1000.0)
                for k in ["locality", "county", "region", "country"]:
                    result[k] = info.get(k, {"_content": ""})["_content"]


    except:
        pass

    return photo_id, result

def _extract_photo_id(url):
    photo_id = None
    if (
            url.find(".static.flickr.com/") > -1
            and (url.find("//farm") > -1 or url.find("//c") > -1)
            and url.find("@") == -1
    ):
        photo_id = int(url.split("/")[-1].split("_")[0])


    return photo_id
