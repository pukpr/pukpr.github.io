/**
 * Dynamically loads images from a GitHub repo directory into a container.
 * If imageDir is omitted or set to "auto", it will be determined from the
 * enclosing HTML file's subdirectory as "results/<subdir>".
 * Adds a caption for each image with the PNG name and a description pulled from the
 * CSV file "sorted_sites_pmsl.csv" in the same directory, where the leading integer
 * in the PNG filename indexes a row in the CSV and the second field is the description.
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

  const apiURL = `https://api.github.com/repos/${owner}/${repo}/contents/${imageDir}`;
  const csvURL = `https://pukpr.github.io/results/sorted_sites_pmsl.csv`;
  const container = document.getElementById(containerId);
  if (!container) return;

  // Helper to parse CSV text into an array of arrays
  function parseCSV(text) {
    // Simple CSV parser for two fields, not handling quoted fields
    return text.split(/\r?\n/).filter(Boolean).map(line => {
      const [index, ...descArr] = line.split(',');
      return [index.trim(), descArr.join(',').trim()];
    });
  }

  try {
    // Fetch and parse CSV file first
    let csvRows = [];
    try {
      const csvRes = await fetch(csvURL);
      if (csvRes.ok) {
        const csvText = await csvRes.text();
        csvRows = parseCSV(csvText); // [[index, desc], ...]
      }
    } catch (e) {
      // ignore if CSV missing
    }
    const descriptions = {};
    csvRows.forEach(([index, desc]) => {
      descriptions[index] = desc;
    });

    // Fetch directory listing from GitHub API
    const res = await fetch(apiURL);
    if (!res.ok) throw new Error("Failed to fetch directory listing.");
    const files = await res.json();
    const imageFiles = files.filter(f => f.type === "file" && /\.(jpg|jpeg|png|gif|webp|svg|png)$/i.test(f.name));
    container.innerHTML = "";

    imageFiles.forEach(f => {
      // Try to extract leading integer from filename (e.g. 0123_blah.png)
      const match = f.name.match(/^(\d+)[_.-]/);
      const idx = match ? match[1] : null;
      const desc = idx && descriptions[idx] ? descriptions[idx] : "";

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
