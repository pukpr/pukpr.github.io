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

### NINO34

![NINO34](nino34/ts.dat-1940-1970.png)

[config](nino34)


### NINO4

![NINO4](nino4/ts.dat-1940-1970.png)

[config](nino4)


{% assign keywords = "nino3,nino12,pdo,emi,1,127,183,245,76,darwin,iod,m6,tna,10,14,202,246,78,denison,iode,nao,tsa,11,155,229,256,1mo,emi,idw,nio12,pdo,111,161,234,41,1o,ic3tsfc,m4,nino3,qbo30" | split: "," %}

{% for kw in keywords %}
### {{ kw | upcase }}

![{{ kw | upcase }}]({{ kw }}/ts.dat-1940-1970.png)

[config]({{ kw }})

{% endfor %}






---

```
$name="2"
$name1=$name + "d"
cp data\$name1.dat .\$name\ts.dat
cp data\$name1.dat.p .\$name\ts.dat.p
cd $name
python3 ..\lte_mlr.py ts.dat --cc --random --low 1940 --high 1970
cd ..
cp nino34\index.md $name
```










