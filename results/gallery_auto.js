/**
 * Dynamically loads images from a GitHub repo directory into a container.
 * If imageDir is omitted or set to "auto", it will be determined from the
 * enclosing HTML file's subdirectory as "results/<subdir>".
 * Adds a caption for each image with the PNG name and a description pulled from
 * the CSV file "sorted_sites_pmsl.csv" in the same directory as this JS file.
 * The CSV's first field is the index (can be negative), and the second is the description.
 * The leading integer in the PNG filename indexes a row in the CSV.
 *
 * @param {Object} opts
 * @param {string} opts.owner - GitHub username/organization
 * @param {string} opts.repo - GitHub repository name
 * @param {string} [opts.imageDir] - Directory inside the repo (e.g. 'images'), or "auto"
 * @param {string} opts.containerId - ID of the DOM element to render images into
 */
async function loadGithubGallery({ owner, repo, imageDir, containerId }) {
  // Auto-detect imageDir if not provided or set to "auto"
  if (!imageDir || imageDir === "auto") {
    let pathParts = window.location.pathname.split('/').filter(Boolean);
    let subdir = pathParts.length > 1 ? pathParts[pathParts.length - 2] : 'unknown';
    imageDir = "results/" + subdir;
  }

  // Construct URLs
  const apiURL = `https://api.github.com/repos/${owner}/${repo}/contents/${imageDir}`;
  // CSV file is in the same directory as gallery.js, so use a relative path
  const jsDir = (() => {
    const scripts = document.getElementsByTagName('script');
    for (let s of scripts) {
      if (s.src && s.src.includes('gallery.js')) {
        // Remove filename, keep path (may be absolute or relative)
        return s.src.slice(0, s.src.lastIndexOf('/') + 1);
      }
    }
    return './'; // fallback
  })();
  const csvURL = jsDir + "sorted_sites_pmsl.csv";
  const container = document.getElementById(containerId);
  if (!container) return;

  // Parse CSV text into an index=>desc map (first field to second field)
  function parseCSV(text) {
    const out = {};
    text.split(/\r?\n/).forEach(line => {
      // Skip empty/comment lines
      if (!line.trim() || line.trim().startsWith("#")) return;
      // Split only the first comma for fields with commas in description
      const match = line.match(/^([^,]+),(.*)$/);
      if (match) {
        const idx = match[1].trim();
        const desc = match[2].trim();
        out[idx] = desc;
      }
    });
    return out;
  }

  // Load CSV file asynchronously (from same origin, as per browser rules)
  async function fetchCSV(url) {
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error('CSV not found');
      const text = await res.text();
      return parseCSV(text);
    } catch (e) {
      return {}; // fallback: no descriptions
    }
  }

  try {
    // Fetch descriptions from CSV
    const descriptions = await fetchCSV(csvURL);

    // Fetch directory listing from GitHub API
    const res = await fetch(apiURL);
    if (!res.ok) throw new Error("Failed to fetch directory listing.");
    const files = await res.json();
    const imageFiles = files.filter(f => f.type === "file" && /\.(jpg|jpeg|png|gif|webp|svg|png)$/i.test(f.name));
    container.innerHTML = "";

    imageFiles.forEach(f => {
      // Try to extract leading integer from filename (e.g. -12_blah.png or 0032-blah.png)
      const match = f.name.match(/^(-?\d+)[_.-]/);
      const idx = match ? match[1] : null;
      const desc = (idx && descriptions[idx]) ? descriptions[idx] : "";

      // Create figure/caption
      const figure = document.createElement("figure");
      figure.style.display = "inline-block";
      figure.style.margin = "8px";
      figure.style.textAlign = "center";

      const img = document.createElement("img");
      img.src = `https://${owner}.github.io/${imageDir}/${encodeURIComponent(f.name)}`;
      img.alt = f.name;
      img.style.maxWidth = "200px";
      img.style.margin = "5px";
      img.style.borderRadius = "5px";

      const figcaption = document.createElement("figcaption");
      figcaption.style.fontSize = "14px";
      figcaption.style.padding = "4px";
      figcaption.innerHTML =
        `<strong>${f.name}</strong>` +
        (desc ? `<br>${desc}` : "");

      figure.appendChild(img);
      figure.appendChild(figcaption);
      container.appendChild(figure);
    });

    if (imageFiles.length === 0) {
      container.textContent = "No images found in /" + imageDir + ".";
    }
  } catch (e) {
    container.textContent = "Failed to load images.";
  }
}
