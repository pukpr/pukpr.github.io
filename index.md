---
layout: default
title: Geoenergy Portal
permalink: /
---

<!-- Previous cirrus attempt: single feTurbulence filter rotated 45° via CSS transform — produced a single wide gray band
<svg xmlns='http://www.w3.org/2000/svg' width='1000' height='1000'>
  <filter id='c' color-interpolation-filters='sRGB'>
    <feTurbulence type='fractalNoise' baseFrequency='0.005 0.08' numOctaves='6' seed='23'/>
    <feColorMatrix type='matrix' values='0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  10 0 0 0 -6'/>
  </filter>
  <rect width='100%' height='100%' filter='url(#c)'/>
</svg>
-->

<!-- Intricate wispy cirrus-cloud overlay: 40+ individual bezier strands in 4 Gaussian-blur layers,
     organised into 7 spatial clusters, all slanted ~45° (lower-left to upper-right). -->
<svg id="cirrus-bg" xmlns="http://www.w3.org/2000/svg"
     style="position:fixed;inset:0;width:100%;height:100%;pointer-events:none;z-index:-1"
     viewBox="0 0 1400 900" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
  <defs>
    <!-- background hazy layer -->
    <filter id="cf-bg"><feGaussianBlur stdDeviation="4 1.2"/></filter>
    <!-- mid-distance layer -->
    <filter id="cf-mid"><feGaussianBlur stdDeviation="2.2 0.6"/></filter>
    <!-- foreground crisp layer -->
    <filter id="cf-fg"><feGaussianBlur stdDeviation="1.2 0.3"/></filter>
    <!-- hair-fine thread layer -->
    <filter id="cf-fine"><feGaussianBlur stdDeviation="0.7 0.2"/></filter>
  </defs>

  <!-- === Layer 1: Background hazy wisps === -->
  <g stroke="white" fill="none" stroke-linecap="round" filter="url(#cf-bg)">
    <!-- Cluster A: upper-left -->
    <path d="M-180,140 C0,120 150,95 320,88 S540,72 720,55"    stroke-width="3"   opacity="0.09"/>
    <path d="M-160,165 C20,145 170,118 340,110 S560,94 740,78"  stroke-width="2"   opacity="0.07"/>
    <path d="M-120,185 C60,165 210,138 380,132 S600,115 780,99" stroke-width="1.8" opacity="0.08"/>
    <path d="M-200,120 C-30,100 120,75 290,68 S510,52 690,35"   stroke-width="2.5" opacity="0.06"/>
    <path d="M-140,200 C40,180 190,155 360,148"                  stroke-width="1.5" opacity="0.07"/>
    <!-- Cluster B: centre-top -->
    <path d="M150,280 C330,260 480,235 650,228 S870,212 1050,195" stroke-width="2.8" opacity="0.09"/>
    <path d="M120,300 C300,280 450,255 620,248 S840,232 1020,215" stroke-width="2"   opacity="0.07"/>
    <path d="M180,260 C360,240 510,215 680,208"                    stroke-width="1.5" opacity="0.08"/>
  </g>

  <!-- === Layer 2: Mid-distance wisps === -->
  <g stroke="white" fill="none" stroke-linecap="round" filter="url(#cf-mid)">
    <!-- Cluster C: upper-right -->
    <path d="M350,50  C530,30  680,5   850,-2 S1070,-18 1250,-35" stroke-width="2.5" opacity="0.12"/>
    <path d="M320,70  C500,50  650,25  820,18 S1040,-2  1220,-19" stroke-width="1.8" opacity="0.10"/>
    <path d="M380,30  C560,10  710,-15 880,-22"                    stroke-width="1.5" opacity="0.11"/>
    <path d="M400,90  C580,70  730,45  900,38 S1120,22 1300,5"    stroke-width="2"   opacity="0.09"/>
    <path d="M430,110 C610,90  760,65  930,58"                     stroke-width="1.2" opacity="0.08"/>
    <!-- Cluster D: left-centre -->
    <path d="M-200,350 C-20,330 130,305 300,298 S520,282 700,265" stroke-width="2.2" opacity="0.11"/>
    <path d="M-180,370 C0,350   150,325 320,318 S540,302 720,285" stroke-width="1.8" opacity="0.09"/>
    <path d="M-160,390 C20,370  170,345 340,338 S560,322 740,305" stroke-width="1.4" opacity="0.10"/>
    <path d="M-220,330 C-40,310 110,285 280,278"                   stroke-width="1"   opacity="0.08"/>
  </g>

  <!-- === Layer 3: Foreground crisp wisps === -->
  <g stroke="white" fill="none" stroke-linecap="round" filter="url(#cf-fg)">
    <!-- Cluster E: lower-left -->
    <path d="M-250,520 C-70,500 80,475  250,468 S470,452 650,435" stroke-width="1.8" opacity="0.12"/>
    <path d="M-220,540 C-40,520 110,495 280,488 S500,472 680,455" stroke-width="1.4" opacity="0.10"/>
    <path d="M-230,555 C-50,535 100,510 270,503"                   stroke-width="1"   opacity="0.09"/>
    <path d="M-270,500 C-90,480 60,455  230,448 S450,432 630,415" stroke-width="2"   opacity="0.08"/>
    <!-- Cluster F: centre -->
    <path d="M300,430 C480,410 630,385 800,378 S1020,362 1200,345" stroke-width="1.8" opacity="0.11"/>
    <path d="M280,450 C460,430 610,405 780,398 S1000,382 1180,365" stroke-width="1.3" opacity="0.09"/>
    <path d="M320,410 C500,390 650,365 820,358"                     stroke-width="1"   opacity="0.10"/>
    <path d="M250,470 C430,450 580,425 750,418"                     stroke-width="0.8" opacity="0.08"/>
    <!-- Cluster G: upper-right extension -->
    <path d="M600,180  C780,160  930,135  1100,128 S1320,112 1400,95"  stroke-width="1.8" opacity="0.12"/>
    <path d="M580,200  C760,180  910,155  1080,148 S1300,132 1420,115" stroke-width="1.4" opacity="0.10"/>
    <path d="M620,160  C800,140  950,115  1120,108"                     stroke-width="1.2" opacity="0.11"/>
    <path d="M640,220  C820,200  970,175  1140,168 S1360,152 1440,135" stroke-width="1"   opacity="0.09"/>
  </g>

  <!-- === Layer 4: Hair-fine thread accents scattered across viewport === -->
  <g stroke="white" fill="none" stroke-linecap="round" filter="url(#cf-fine)">
    <path d="M-150,240 C30,220  180,195 350,188 S570,172 750,155"     stroke-width="0.8" opacity="0.10"/>
    <path d="M-100,255 C80,235  230,210 400,203"                       stroke-width="0.6" opacity="0.08"/>
    <path d="M50,320   C230,300 380,275 550,268 S770,252 950,235"      stroke-width="0.8" opacity="0.09"/>
    <path d="M500,580  C680,560 830,535 1000,528 S1220,512 1400,495"  stroke-width="0.7" opacity="0.10"/>
    <path d="M480,600  C660,580 810,555 980,548"                       stroke-width="0.5" opacity="0.08"/>
    <path d="M700,640  C880,620 1030,595 1200,588 S1380,572 1440,560" stroke-width="0.8" opacity="0.09"/>
    <path d="M100,480  C280,460 430,435 600,428 S820,412 1000,395"    stroke-width="0.6" opacity="0.08"/>
    <path d="M-50,460  C130,440 280,415 450,408"                       stroke-width="0.7" opacity="0.09"/>
    <path d="M900,280  C1080,260 1230,235 1400,228"                    stroke-width="0.8" opacity="0.10"/>
    <path d="M850,300  C1030,280 1180,255 1350,248"                    stroke-width="0.6" opacity="0.08"/>
    <path d="M200,680  C380,660 530,635 700,628 S920,612 1100,595"    stroke-width="0.7" opacity="0.09"/>
    <path d="M-100,720 C80,700  230,675 400,668"                       stroke-width="0.8" opacity="0.08"/>
    <path d="M750,760  C930,740 1080,715 1250,708"                     stroke-width="0.5" opacity="0.07"/>
    <path d="M1050,350 C1150,330 1280,310 1400,300"                    stroke-width="0.7" opacity="0.09"/>
    <path d="M-300,620 C-120,600 30,575 200,568"                       stroke-width="0.6" opacity="0.08"/>
  </g>
</svg>

<!-- Light-theme cartography map: Boundary Waters / Quetico style topo map background -->
<svg id="map-bg" xmlns="http://www.w3.org/2000/svg"
     style="position:fixed;inset:0;width:100%;height:100%;pointer-events:none;z-index:-1"
     viewBox="0 0 1400 900" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
  <defs>
    <!-- Fine grid pattern mimicking topo-map UTM grid -->
    <pattern id="topo-grid" width="70" height="70" patternUnits="userSpaceOnUse">
      <path d="M 70 0 L 0 0 0 70" fill="none" stroke="rgba(80,80,80,0.13)" stroke-width="0.5"/>
    </pattern>
    <!-- Forest hatch texture -->
    <pattern id="forest-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
      <line x1="0" y1="0" x2="0" y2="6" stroke="rgba(80,110,60,0.18)" stroke-width="1"/>
    </pattern>
  </defs>

  <!-- === Land base (parchment-green, like printed topo paper) === -->
  <rect width="1400" height="900" fill="#eaede5"/>

  <!-- === Forest / upland patches (slightly deeper green) === -->
  <path d="M0,0 C120,20 280,10 420,40 C560,70 650,30 780,0 L1400,0 L1400,320
           C1280,300 1150,330 1050,310 C920,285 820,340 700,320
           C580,300 460,350 340,330 C220,310 100,360 0,340 Z"
        fill="#d8e2ce" opacity="0.7"/>
  <path d="M0,900 L0,600 C80,620 180,590 280,610 C380,630 480,600 580,615
           C680,630 760,600 860,590 C960,580 1060,610 1160,595
           C1260,580 1340,600 1400,590 L1400,900 Z"
        fill="#d8e2ce" opacity="0.6"/>
  <path d="M700,200 C750,180 840,190 900,220 C960,250 980,300 1040,310
           C1100,320 1180,290 1220,260 C1260,230 1300,240 1340,270
           L1400,280 L1400,450 C1340,440 1260,460 1180,450
           C1100,440 1020,470 940,460 C860,450 780,480 700,460
           C620,440 560,420 520,380 C480,340 500,280 540,250 C580,220 650,220 700,200 Z"
        fill="#d4deca" opacity="0.65"/>

  <!-- === Water bodies (light steel-blue, organic lake shapes) === -->

  <!-- Quetico Lake – large, lower-left, elongated E-W -->
  <path d="M0,660 C30,640 70,635 110,648 C150,661 175,650 210,642
           C245,634 275,640 310,655 C345,670 370,660 400,648
           C430,636 455,645 480,658 C505,671 520,680 510,700
           C500,720 475,735 445,742 C415,749 385,740 355,750
           C325,760 295,775 265,770 C235,765 205,780 175,775
           C145,770 115,785 85,778 C55,771 25,760 0,750 Z"
        fill="#bdd4e6" opacity="0.88"/>
  <path d="M0,750 C40,755 80,768 120,772 C160,776 195,762 230,770
           C265,778 300,790 335,785 C370,780 405,765 430,775
           C455,785 470,800 460,820 C450,840 420,850 380,858
           C340,866 290,860 250,870 C210,880 170,895 130,900
           L0,900 Z"
        fill="#bdd4e6" opacity="0.80"/>

  <!-- Jean Lake – lower-centre -->
  <path d="M520,720 C555,705 600,700 640,712 C680,724 710,718 745,705
           C780,692 815,698 845,715 C875,732 890,755 875,772
           C860,789 830,795 800,790 C770,785 745,795 715,800
           C685,805 655,798 625,790 C595,782 565,790 540,778
           C515,766 505,745 520,720 Z"
        fill="#bdd4e6" opacity="0.85"/>

  <!-- Kawnipi / upper-right lake complex -->
  <path d="M920,280 C955,265 1000,260 1045,272 C1090,284 1120,278 1155,262
           C1190,246 1230,252 1260,270 C1290,288 1310,310 1295,332
           C1280,354 1250,360 1215,355 C1180,350 1155,365 1125,375
           C1095,385 1060,378 1030,362 C1000,346 975,355 950,345
           C925,335 905,315 920,280 Z"
        fill="#bdd4e6" opacity="0.88"/>
  <path d="M1100,360 C1130,350 1165,348 1200,360 C1235,372 1265,368 1295,355
           L1340,348 C1370,342 1400,348 1400,348 L1400,430
           C1370,435 1340,425 1310,430 C1280,435 1255,450 1220,445
           C1185,440 1160,455 1130,448 C1100,441 1075,425 1070,405
           C1065,385 1070,370 1100,360 Z"
        fill="#bdd4e6" opacity="0.82"/>

  <!-- Cirque Lake – left-centre -->
  <path d="M65,430 C95,418 130,415 165,428 C200,441 225,435 255,420
           C285,405 320,410 345,428 C370,446 375,470 358,488
           C341,506 310,510 280,504 C250,498 225,512 195,518
           C165,524 135,515 110,500 C85,485 60,470 65,430 Z"
        fill="#bdd4e6" opacity="0.85"/>

  <!-- Batchewaung / upper-left lake -->
  <path d="M0,180 C35,165 75,160 110,175 C145,190 170,182 200,168
           C230,154 262,160 285,178 C308,196 312,222 295,238
           C278,254 250,257 220,250 C190,243 165,255 135,262
           C105,269 70,260 40,248 C10,236 -5,215 0,180 Z"
        fill="#bdd4e6" opacity="0.80"/>

  <!-- Numerous small lakes / ponds (scattered) -->
  <ellipse cx="460" cy="160" rx="48" ry="22" fill="#bdd4e6" opacity="0.75"/>
  <ellipse cx="580" cy="240" rx="35" ry="18" fill="#bdd4e6" opacity="0.72"/>
  <ellipse cx="670" cy="130" rx="55" ry="20" fill="#bdd4e6" opacity="0.75"/>
  <ellipse cx="790" cy="195" rx="42" ry="25" fill="#bdd4e6" opacity="0.72"/>
  <ellipse cx="860" cy="110" rx="38" ry="16" fill="#bdd4e6" opacity="0.70"/>
  <ellipse cx="1000" cy="160" rx="50" ry="22" fill="#bdd4e6" opacity="0.75"/>
  <ellipse cx="1110" cy="210" rx="30" ry="15" fill="#bdd4e6" opacity="0.70"/>
  <ellipse cx="350" cy="520" rx="40" ry="19" fill="#bdd4e6" opacity="0.75"/>
  <ellipse cx="460" cy="555" rx="28" ry="14" fill="#bdd4e6" opacity="0.70"/>
  <ellipse cx="620" cy="490" rx="45" ry="20" fill="#bdd4e6" opacity="0.72"/>
  <ellipse cx="730" cy="560" rx="36" ry="17" fill="#bdd4e6" opacity="0.72"/>
  <ellipse cx="840" cy="490" rx="52" ry="22" fill="#bdd4e6" opacity="0.75"/>
  <ellipse cx="960" cy="540" rx="38" ry="18" fill="#bdd4e6" opacity="0.72"/>
  <ellipse cx="1060" cy="480" rx="44" ry="20" fill="#bdd4e6" opacity="0.74"/>
  <ellipse cx="1180" cy="510" rx="32" ry="16" fill="#bdd4e6" opacity="0.70"/>
  <ellipse cx="1290" cy="470" rx="48" ry="21" fill="#bdd4e6" opacity="0.73"/>
  <ellipse cx="200" cy="310" rx="35" ry="16" fill="#bdd4e6" opacity="0.70"/>
  <ellipse cx="310" cy="260" rx="28" ry="13" fill="#bdd4e6" opacity="0.68"/>
  <ellipse cx="420" cy="340" rx="40" ry="18" fill="#bdd4e6" opacity="0.72"/>
  <ellipse cx="580" cy="370" rx="30" ry="14" fill="#bdd4e6" opacity="0.70"/>
  <ellipse cx="700" cy="390" rx="44" ry="18" fill="#bdd4e6" opacity="0.73"/>
  <ellipse cx="1320" cy="620" rx="50" ry="22" fill="#bdd4e6" opacity="0.75"/>
  <ellipse cx="1200" cy="660" rx="36" ry="16" fill="#bdd4e6" opacity="0.70"/>
  <ellipse cx="1080" cy="620" rx="42" ry="18" fill="#bdd4e6" opacity="0.72"/>
  <ellipse cx="950" cy="650" rx="30" ry="14" fill="#bdd4e6" opacity="0.68"/>

  <!-- === Topographic contour lines (subtle brown-grey curves) === -->
  <g fill="none" stroke="rgba(100,80,55,0.22)" stroke-width="0.8" stroke-linecap="round">
    <path d="M0,80 C200,60 400,90 600,70 C800,50 1000,80 1200,65 C1300,58 1360,72 1400,68"/>
    <path d="M0,120 C180,100 360,128 560,108 C760,88 960,118 1160,102 C1280,93 1360,108 1400,104"/>
    <path d="M0,480 C160,460 340,490 520,472 C700,454 880,482 1060,465 C1200,452 1320,468 1400,462"/>
    <path d="M0,560 C140,542 320,568 500,552 C680,536 860,562 1040,547 C1180,535 1300,549 1400,545"/>
    <path d="M200,320 C320,305 460,328 600,312 C740,296 880,320 1020,306 C1140,294 1260,310 1380,304"/>
    <path d="M100,380 C240,363 400,388 560,372 C720,356 900,380 1060,365 C1200,352 1320,367 1400,363"/>
    <path d="M0,820 C180,804 380,828 580,812 C780,796 980,820 1180,806 C1300,797 1360,808 1400,806"/>
    <path d="M0,860 C200,846 420,866 640,852 C860,838 1080,858 1300,846 L1400,844"/>
  </g>

  <!-- === Border dashes (US/Canada boundary) === -->
  <path d="M0,95 C200,80 450,100 700,88 C950,76 1200,95 1400,85"
        fill="none" stroke="rgba(180,30,30,0.55)" stroke-width="1.2"
        stroke-dasharray="6,4"/>

  <!-- === UTM grid overlay === -->
  <rect width="1400" height="900" fill="url(#topo-grid)"/>

  <!-- === Map label (very faint, mimics printed topo sheet text) === -->
  <text x="340" y="640" font-family="serif" font-size="22" font-weight="bold"
        fill="rgba(60,60,60,0.14)" letter-spacing="6" text-anchor="middle"
        transform="rotate(-2,340,640)">Q U E T I C O</text>
  <text x="700" y="665" font-family="serif" font-size="18"
        fill="rgba(60,60,60,0.12)" letter-spacing="4" text-anchor="middle"
        transform="rotate(-2,700,665)">P R O V I N C I A L</text>
  <text x="1050" y="640" font-family="serif" font-size="22" font-weight="bold"
        fill="rgba(60,60,60,0.14)" letter-spacing="6" text-anchor="middle"
        transform="rotate(-2,1050,640)">P A R K</text>
  <text x="160" y="745" font-family="serif" font-size="15"
        fill="rgba(40,60,100,0.22)" letter-spacing="3" text-anchor="middle"
        transform="rotate(-4,160,745)">QUETICO LAKE</text>
  <text x="670" y="778" font-family="serif" font-size="13"
        fill="rgba(40,60,100,0.20)" letter-spacing="2" text-anchor="middle"
        transform="rotate(-2,670,778)">JEAN LAKE</text>
  <text x="1095" y="330" font-family="serif" font-size="13"
        fill="rgba(40,60,100,0.20)" letter-spacing="2" text-anchor="middle"
        transform="rotate(-3,1095,330)">KAWNIPI LAKE</text>
</svg>

<section class="hero" id="top" style="background-image:url('/assets/media/bg_villars.jpg'); background-size:cover; background-position:center; background-repeat:no-repeat;">
  <div class="hero-grid">
    <div class="hero-copy">
      <span class="eyebrow">Scientific workbench</span>
      <h1>G.E.M.</h1>
      <div class="hero-stats" aria-label="Portal overview">
        <div class="stat"><strong>8</strong><span>Primary destinations</span></div>
        <div class="stat"><strong>10+</strong><span>Interactive modules</span></div>
        <div class="stat"><strong>6</strong><span>Research artifacts</span></div>
      </div>
    </div>
    <aside class="hero-panel" aria-label="Portal principles">
      <span class="card-kicker">spanning</span>
      <h3>GeoEnergyMath by Paul Pukite</h3>
      <p></p>
    </aside>
  </div>
</section>

<section aria-labelledby="destinations-title">
  <div class="section-head">
    <div>
      <span class="eyebrow">Destinations</span>
      <h2 id="destinations-title">primary</h2>
    </div>
    <p>climate, tides, energy</p>
  </div>

  <div class="portal-grid">
    <article class="card featured">
      <div>
        <span class="card-kicker">Featured research</span>
        <h3>Climate indices and sea-level modeling</h3>
        <p>models and examples => <i> ENSO, QBO, MSL, NAO, etc </i><br/>
        <a href="lte-whitepaper.html">White paper on Lunisolar Common-Mode forcing</a>
        </p>
      </div>
      <div class="link-list">
        <div class="link-item">
          <div class="link-cluster">
            <a href="results/image_results.html">Climate Indices and PSMSL Stations model</a>
            <div class="sub-links">
              <a href="examples/warne_intro.html">Warnemunde intro</a>
              <a href="GEM-LTE/gem-lte-results.html">Cross-validation results</a>
              <a href="examples/pysindy/">PySINDy latent layer example</a>
              <a href="examples/mlr/">Multiple linear regression example</a>
            </div>
          </div>
          <span class="link-meta">Model hub</span>
        </div>
      </div>
    </article>

    <article class="card">
      <div>
        <span class="card-kicker">Interactive</span>   
        <h3>Hands-on models</h3>
        <p>browser-based simulations to explore, not just reading.</p>
      </div>
      <div class="link-list">
        <div class="link-item"><a href="ChandlerWobble">Interactive Chandler wobble model</a><span class="link-meta">Simulation</span></div>
        <div class="link-item"><a href="OilShockModel/oil-shock-model">Interactive Oil Shock Model</a><span class="link-meta">Scenario tool</span></div>
        <div class="link-item"><a href="GEM-LTE/pukite-qbo">Interactive QBO Model</a><span class="link-meta">Simulation</span></div>
        <div class="link-item"><a href="GEM-LTE/gem-lte-results">Interactive LTE comparison</a><br/>
                               <a href="GEM-LTE/pukite-slr">Interactive LTE modeler</a><br/>
                               <a href="GEM-LTE/lte-signature-map">LTE sig</a><span class="link-meta">Simulation</span></div>
        <div class="link-item"><a href="GEM-LTE/pukite-lod">Interactive LOD Model</a><span class="link-meta">Simulation</span></div>
        <div class="link-item"><a href="AdaPACE/earth-science-visualization" target="_blank" rel="noopener noreferrer">AdaPACE Earth-science visualization examples</a><span class="link-meta">Docs</span></div>
        <div class="link-item">
            <a href="https://github.com/pukpr/context" target="_blank" rel="noopener noreferrer">Earth sciences context modeling & knowledgebase server</a>
            <div class="sub-links">
              <a href="https://geoenergymath.com/wp-content/uploads/2020/06/a4adf-d-knowledge_based_enviromental_modeling.compressed.pdf" target="_blank" rel="noopener noreferrer">White paper </a>
            </div>
        </div>
      </div>
     
    </article>

    <article class="card">
      <div>
        <span class="card-kicker">Writing</span>
        <h3>Blogs and publications</h3>
        <p>background material</p>
      </div>
      <div class="link-list">
        <div class="link-item"><a href="https://GeoEnergyMath.com" target="_blank" rel="noopener noreferrer">GeoEnergyMath blog</a><span class="link-meta">Posts</span></div>
        <div class="link-item"><a href="https://agupubs.onlinelibrary.wiley.com/doi/book/10.1002/9781119434351" target="_blank" rel="noopener noreferrer">Mathematical Geoenergy book</a><span class="link-meta">Book</span></div>
        <div class="link-item"><a href="https://PeakOilBarrel.com" target="_blank" rel="noopener noreferrer">Peak Oil Barrel depletion modeling blog</a><span class="link-meta">Posts</span></div>
        <div class="link-item"><a href="https://github.com/orgs/azimuth-project/discussions" target="_blank" rel="noopener noreferrer">Azimuth Project forum</a><span class="link-meta">Discussion</span></div>
        <div class="link-item"><a href="https://pukite.substack.com/" target="_blank" rel="noopener noreferrer">SubSurface substack</a><span class="link-meta">Blog</span>
            <div class="sub-links">
              <a href="https://medium.com/@puk_54065" target="_blank" rel="noopener noreferrer">Medium</a>
            </div>
        </div>
        <div class="link-item"><a href="https://scholar.google.com/citations?hl=en&user=J4XWUG8AAAAJ" target="_blank" rel="noopener noreferrer">Google Scholar</a><span class="link-meta">Paper Publications</span></div>
        <div class="link-item"><a href="https://www.researchgate.net/publication/283579370_C2M2L_Final_Report" target="_blank" rel="noopener noreferrer">C2M2L environment modeling via ontological knowledgebases (2013)</a><span class="link-meta">Report</span></div>
        <div class="link-item">
          <div class="link-cluster">
            <a href="https://scholar.google.com/citations?view_op=view_citation&hl=en&oe=ASCII&user=J4XWUG8AAAAJ&citation_for_view=J4XWUG8AAAAJ:-f6ydRqryjwC" target="_blank" rel="noopener noreferrer">ESD Ideas: long-period tidal forcing in geophysics (2020)</a>
            <a href="https://scholar.google.com/citations?view_op=view_citation&hl=en&oe=ASCII&user=J4XWUG8AAAAJ&citation_for_view=J4XWUG8AAAAJ:W7OEmFMy1HYC" target="_blank" rel="noopener noreferrer">Sloshing Model for ENSO (2014)</a>
          </div>
          <span class="link-meta">Preprints</span>
        </div>
      </div>
    </article>


    <article class="card">
      <div>
        <span class="card-kicker">Models</span>
        <h3>Repositories and technical notes</h3>
        <p>source code, docs, and gists</p>
      </div>
      <div class="link-list">
        <div class="link-item">
          <div class="link-cluster">
            <a href="https://github.com/pukpr/GEM-LTE" target="_blank" rel="noopener noreferrer">GEM LTE model source repository</a>
            <div class="sub-links">
              <a href="https://gist.github.com/pukpr" target="_blank" rel="noopener noreferrer">GitHub gists</a>
            </div>
            <a href="https://github.com/pukpr/OilShockModel" target="_blank" rel="noopener noreferrer">Oil Shock Model source repository</a>
            <div class="sub-links">
              <a href="https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1002/9781119434351.app2" target="_blank" rel="noopener noreferrer">Mathematical Geoenergy appendix</a>
            </div>
            <a href="https://github.com/pukpr/ChandlerWobble" target="_blank" rel="noopener noreferrer">Chandler wobble model source repository</a>
            
            <a href="https://github.com/pukpr/LTEsparse" target="_blank" rel="noopener noreferrer">LTE in Julia source repository</a>
          </div>
          <span class="link-meta">Source</span>
        </div>
        <div class="link-item"><a href="gem/models">Model derivations and source code from chapters</a>
          <span class="link-meta">Docs</span></div>
        </div>
    </article>


    <article class="card">
      <div>
        <span class="card-kicker">Software</span>
        <h3>Libraries and applications</h3>
        <p>Building and analysis of reliable distributed systems</p>
      </div>
      <div class="link-list">
        <div class="link-item">
          <div class="link-cluster">
            <a href="https://github.com/pukpr/AdaPACE" target="_blank" rel="noopener noreferrer">Real-time distributed simulation and communication library</a>
            <div class="sub-links">
              <a href="https://github.com/pukpr/AdaPACE/tree/main/examples" target="_blank" rel="noopener noreferrer">Visualization examples</a>
              <a href="https://scholar.google.com/citations?view_op=view_citation&hl=en&oe=ASCII&user=J4XWUG8AAAAJ&citation_for_view=J4XWUG8AAAAJ:5nxA0vEk-isC" target="_blank" rel="noopener noreferrer">SUMIT virtual reality toolkit (2005)</a>
            </div>
            <a href="https://github.com/pukpr/degas" target="_blank" rel="noopener noreferrer">Multitasking simulation library</a>
            <div class="sub-links">
              <a href="https://dl.acm.org/doi/abs/10.1145/1185875.1185649" target="_blank" rel="noopener noreferrer">Generic discrete event simulations using DEGAS (2007)</a>
              <a href="https://scholar.google.com/citations?view_op=view_citation&hl=en&oe=ASCII&user=J4XWUG8AAAAJ&citation_for_view=J4XWUG8AAAAJ:d1gkVwhDpl0C" target="_blank" rel="noopener noreferrer">DEGAS: discrete event Gnu advanced scheduler (2006)</a>
            </div>
            <a href="https://github.com/pukpr/CARMS" target="_blank" rel="noopener noreferrer">CARMS Markov model analysis</a>
            <div class="sub-links">
              <a href="https://www.wiley.com/en-au/Modeling+for+Reliability+Analysis%3A+Markov+Modeling+for+Reliability%2C+Maintainability%2C+Safety%2C+and+Supportability+Analyses+of+Complex+Systems-p-9780780334823" target="_blank" rel="noopener noreferrer">Modeling for Reliability Analysis</a>
            </div>
          </div>
          <span class="link-meta">Source</span>
        </div>
      </div>
    </article>

    <article class="card">
      <div>
        <span class="card-kicker">Context</span>
        <h3>AI, context modeling, and Earth sciences</h3>
        <p>knowledge-based environmental modeling bridging systems research and geophysics</p>
      </div>
      <div class="link-list">
        <div class="link-item">
          <div class="link-cluster">
            <a href="https://github.com/pukpr/context" target="_blank" rel="noopener noreferrer">Earth sciences context modeling & knowledgebase server</a>
            <div class="sub-links">
              <a href="https://geoenergymath.com/wp-content/uploads/2020/06/a4adf-d-knowledge_based_enviromental_modeling.compressed.pdf" target="_blank" rel="noopener noreferrer">Knowledge-based environmental modeling white paper</a>
            </div>
            <a href="https://www.researchgate.net/publication/283579370_C2M2L_Final_Report" target="_blank" rel="noopener noreferrer">C2M2L environment modeling via ontological knowledgebases (2013)</a>
            <div class="sub-links">
              <a href="https://scholar.google.com/citations?view_op=view_citation&hl=en&oe=ASCII&user=J4XWUG8AAAAJ&citation_for_view=J4XWUG8AAAAJ:UeHWp8X0CEIC" target="_blank" rel="noopener noreferrer">Knowledge-Based Environmental Context Modeling (2012)</a>
              <a href="https://scholar.google.com/citations?view_op=view_citation&hl=en&oe=ASCII&user=J4XWUG8AAAAJ&citation_for_view=J4XWUG8AAAAJ:_kc_bZDykSQC" target="_blank" rel="noopener noreferrer">Stochastic Analysis for Context Modeling (2012)</a>
              <a href="https://scholar.google.com/citations?view_op=view_citation&hl=en&oe=ASCII&user=J4XWUG8AAAAJ&citation_for_view=J4XWUG8AAAAJ:4TOpqqG69KYC" target="_blank" rel="noopener noreferrer">Unified characterization of surface topography for vehicle dynamics applications (2012)</a>
              <a href="https://scholar.google.com/citations?view_op=view_citation&hl=en&oe=ASCII&user=J4XWUG8AAAAJ&citation_for_view=J4XWUG8AAAAJ:ULOm3_A8WrAC" target="_blank" rel="noopener noreferrer">Characterizing diffusive growth by uncertainty quantification (2012)</a>
            </div>
            <a href="https://GeoEnergyMath.com" target="_blank" rel="noopener noreferrer">GeoEnergyMath blog</a>
          </div>
          <span class="link-meta">Bridge work</span>
        </div>
      </div>
    </article>

    <article class="card">
      <div>
        <span class="card-kicker">Conference</span>
        <h3>Presentations</h3>
        <p>AGU, EGU, and ICLR presentation record</p>
      </div>
      <div class="link-list">
        <div class="link-item"><a href="https://scholar.google.com/citations?view_op=view_citation&hl=en&oe=ASCII&user=J4XWUG8AAAAJ&citation_for_view=J4XWUG8AAAAJ:9ZlFYXVOiuMC" target="_blank" rel="noopener noreferrer">Analytical formulation of equatorial standing wave phenomena: Application to QBO and ENSO</a><span class="link-meta">AGU 2016</span></div>
        <div class="link-item">
          <div class="link-cluster">
            <a href="https://www.authorea.com/doi/full/10.1002/essoar.b1c62a3df907a1fa.b18572c23dc245c9" target="_blank" rel="noopener noreferrer">Biennial-Aligned Lunisolar-Forcing of ENSO</a>
            <a href="https://agu.confex.com/agu/fm17/meetingapp.cgi/Paper/276374" target="_blank" rel="noopener noreferrer">Knowledge-Based Environmental Context Modeling</a>
            <div class="sub-links">
              <a href="https://i0.wp.com/imageshack.com/a/img923/7645/yKKjgL.png" target="_blank" rel="noopener noreferrer">Poster</a>
            </div>
          </div>
          <span class="link-meta">AGU 2017</span>
        </div>
        <div class="link-item">
          <div class="link-cluster">
             <a href="https://scholar.google.com/citations?view_op=view_citation&hl=en&oe=ASCII&user=J4XWUG8AAAAJ&citation_for_view=J4XWUG8AAAAJ:2osOgNQ5qMEC" target="_blank" rel="noopener noreferrer">Ephemeris calibration of Laplace's tidal equation model for ENSO</a>
             <a href="https://scholar.google.com/citations?view_op=view_citation&hl=en&oe=ASCII&user=J4XWUG8AAAAJ&citation_for_view=J4XWUG8AAAAJ:JV2RwH3_ST0C" target="_blank" rel="noopener noreferrer">An interactive framework for understanding mathematical geoenergy</a>
             <a href="https://scholar.google.com/citations?view_op=view_citation&hl=en&oe=ASCII&user=J4XWUG8AAAAJ&citation_for_view=J4XWUG8AAAAJ:Se3iqnhoufwC" target="_blank" rel="noopener noreferrer">Permian Basin Tight Oil Model to Predict Future Output</a>
          </div>
          <span class="link-meta">AGU 2018</span>
        </div>
        <div class="link-item">
          <div class="link-cluster">
            <a href="https://meetingorganizer.copernicus.org/EGU21/EGU21-10515.html" target="_blank" rel="noopener noreferrer">Nonlinear long-period tidal forcing with application to ENSO, QBO, and Chandler wobble</a>
            <div class="sub-links">
              <a href="https://geoenergymath.com/wp-content/uploads/2021/04/supplementarydocument-egu-2021.pdf" target="_blank" rel="noopener noreferrer">Supplementary document</a>
            </div>
          </div>
          <span class="link-meta">EGU 2021</span>
        </div>
        <div class="link-item"><a href="https://openreview.net/forum?id=XqOseg0L9Q" target="_blank" rel="noopener noreferrer">Nonlinear Differential Equations with external forcing</a><span class="link-meta">ICLR 2020</span></div>
      </div>
    </article>


    <article class="card">
      <div>
        <span class="card-kicker">Related</span>
        <h3>External project links</h3>
        <p>adjacent sites</p>
      </div>
      <div class="link-list">
        <div class="link-item"><a href="https://azimuth-project.github.io" target="_blank" rel="noopener noreferrer">Azimuth Project climate modeling</a><span class="link-meta">Collaboration site</span></div>
        <div class="link-item"><a href="https://www.realclimate.org/" target="_blank" rel="noopener noreferrer">Real Climate </a><span class="link-meta">Blog & Comments</span>
          <div class="sub-links">
             <a href="rc.md" target="_blank" rel="noopener noreferrer">meta analysis for @whut</a>
          </div>
        </div>
        <div class="link-item"><a href="https://twitter.com/whut" target="_blank" rel="noopener noreferrer">@WHUT</a><span class="link-meta">Twitter</span></div>
        <div class="link-item"><a href="https://bsky.app/profile/pukite.com" target="_blank" rel="noopener noreferrer">@pukite.com</a><span class="link-meta">BlueSky</span>
          <div class="sub-links">
             <a href="https://bsky.app/profile/did:plc:2ge3m52s47jdba642nmukk3q/feed/peakoil" target="_blank" rel="noopener noreferrer">Peak Oil feed</a>
             <a href="https://bsky.app/profile/did:plc:2ge3m52s47jdba642nmukk3q/feed/climate_cycles" target="_blank" rel="noopener noreferrer">Climate Cycle feed</a>
             <a href="https://bsky.app/profile/did:plc:2ge3m52s47jdba642nmukk3q/feed/copernicus" target="_blank" rel="noopener noreferrer">Symbolic Regression feed</a>
          </div>
        </div>
      </div>
    </article>

    <article class="card">
      <div>
        <span class="card-kicker">Media</span>
        <h3>Explanatory videos</h3>
        <p>recorded simulations</p>
      </div>
      <div class="link-list">
        <div class="link-item"><a href="https://youtu.be/KHX6xBEcUcU" target="_blank" rel="noopener noreferrer">QBO</a><span class="link-meta">YouTube</span></div>
        <div class="link-item"><a href="https://www.youtube.com/shorts/KSy8VkXizhs" target="_blank" rel="noopener noreferrer">Chandler wobble</a><span class="link-meta">YouTube</span></div>
        <div class="link-item"><a href="https://youtu.be/MjhVm-Yz9XI" target="_blank" rel="noopener noreferrer">Laplace's tidal modulation</a><span class="link-meta">YouTube</span></div>
      </div>
    </article>
  
  </div>
</section>


<section class="artifacts" id="artifacts" aria-labelledby="artifacts-title">
  <div class="section-head">
    <div>
      <span class="eyebrow">Archive</span>
      <h2 id="artifacts-title">Artifacts</h2>
    </div>
    <p>artifacts archive</p>
  </div>
  <div class="artifact-list">
    <div class="artifact-item">
      <div class="artifact-index">1</div>
        <a href="https://x-server.gmca.aps.anl.gov/TRDS_sl.html">Surface scattering</a>
        <span>Analysis site</span>
     </div>
    <div class="artifact-item">
      <div class="artifact-index">2</div>
      <a href="The modelled climatic response to the 18.6-year lunar nodal cycle and its role in decadal temperature trends.pdf">The modelled climatic response to the 18.6-year lunar nodal cycle and its role in decadal temperature trends</a>
      <span>Review paper</span>
      </div>
    <div class="artifact-item">
      <div class="artifact-index">3</div>
      <a href="mathematical-geoenergy-findings/">Mathematical Geoenergy revisited: review of the major findings and their likely significance</a>
      <span>Review paper</span>
      </div>
    <div class="artifact-item">
      <div class="artifact-index">4</div>
      <a href="https://geoenergymath.com/2020/03/18/mathematical-geoenergy-2/" target="_blank" rel="noopener noreferrer">Mathematical Geoenergy findings list</a>
      <span>Archive index</span>
      </div>
    <div class="artifact-item">
      <div class="artifact-index">5</div>
      <a href="https://geoenergymath.com/2026/04/11/gem-lte-modeling/" target="_blank" rel="noopener noreferrer">February 2026 GEM-LTE mean sea level and climate index cross-validation archive</a>
      <span>Results archive</span>
      </div>
    </div>
</section>
