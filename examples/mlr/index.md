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

<p><i>
=> {% include_relative nino34/label.txt %}
</i></p>

![NINO34](nino34/ts.dat-1940-1970.png)

[config](nino34)


{% assign keywords = "nino4,1,127,183,245,76,darwin,iod,m6,tna,10,14,202,246,78,denison,iode,nao,tsa,11,155,229,256,amo,emi,iodw,nino12,pdo,111,161,234,41,ao,ic3tsfc,m4,nino3,qbo30" | split: "," %}

{% for kw in keywords %}
#### {{ kw | upcase }}
<p><i>
=> {% include_relative {{ kw }}/label.txt %}
</i></p>

![{{ kw | upcase }}]({{ kw }}/ts.dat-1940-1970.png)

[config]({{ kw }})

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










