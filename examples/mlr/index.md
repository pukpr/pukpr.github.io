# LTE multiple linear regression on latent layers





---

## 📖 About



---

## 🛠️ How to Use





1. Clone this repository:



```
git clone https://github.com/pukpr/pukpr.github.io

$env:max_iters=3
$env:align=1
$env:PYPATH=".."
python3 ..\lte_mlr.py ts.dat --cc --random --low 1940 --high 1970

```


---

{% assign keywords = "nino34,nino4,1,127,183,245,76,darwin,iod,m6,tna,10,14,202,246,78,denison,iode,nao,tsa,11,155,225,154,229,256,amo,emi,iodw,nino12,pdo,111,161,234,41,ao,ic3tsfc,m4,nino3,qbo30,7,8,42,113,119,172,179,194,2,20,203,22,23,236,239,24,240,249,25,285,302,32,33,330,47,5,57,58,62,68,69,70,71,72,73,79,80,81,82,88,89,91,95,98" | split: "," %}

{% for kw in keywords %}
#### {{ kw | upcase }}
{% assign key = kw | strip | append: "" %}

*description:* {% include_relative {{ kw }}/label.txt %} => {{ site.data.ID[key].Name }} in {{ site.data.ID[key].Country }}

![{{ kw | upcase }}]({{ kw }}/ts.dat-1940-1970.png)


[config]({{ kw }})  | [location](../maps/{{ kw }}_loc.png)

{% endfor %}




---

```
$name="183"
$name1=$name + "d"
cp data\$name1.dat .\$name\ts.dat
cp data\$name1.dat.p .\$name\ts.dat.p
cd $name
python3 ..\lte_mlr.py ts.dat --cc --random --low 1940 --high 1970
cd ..
cp nino34\index.md $name
```

---

{% if site.data.ID %}
  <ul>
  {% for pair in site.data.ID %}
    {% assign key = pair[0] %}
    {% assign rec = pair[1] %}
    <li>{{ key }} — {{ rec.Name }} ({{ rec.Country }})</li>
  {% endfor %}
  </ul>
{% else %}
  <p>site.data.ID not found</p>
{% endif %}



```
python3 ..\..\pysindy\plot_lte_scatter.py --header --out lte.png --lo 1940 --high 1970
python3 ..\..\..\..\python\plot_sinusoids_from_json_with_bars.py --out-ts ts-sin.png --out-bars ts-bar.png  --start 1920-01  --end 2030-01 ..\warne.dat.p ts.dat.p
python3 ..\..\pysindy\cc.py --low 1940 --high 1970

```





