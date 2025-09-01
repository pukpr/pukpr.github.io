/**
 * Dynamically loads images from a GitHub repo directory into a container.
 * If imageDir is omitted or set to "auto", it will be determined from the
 * enclosing HTML file's subdirectory as "results/<subdir>".
 * @param {Object} opts
 * @param {string} opts.owner - GitHub username/organization
 * @param {string} opts.repo - GitHub repository name
 * @param {string} [opts.imageDir] - Directory inside the repo (e.g. 'images'), or "auto"
 * @param {string} opts.containerId - ID of the DOM element to render images into
 */
async function loadGithubGallery({ owner, repo, imageDir, containerId }) {
  // Auto-detect imageDir if not provided or set to "auto"
  if (!imageDir || imageDir === "auto") {
    // e.g. /results/ci_pt5_quad/index.html → ['results', 'ci_pt5_quad', 'index.html']
    let pathParts = window.location.pathname.split('/').filter(Boolean);
    // The subdir is the last part before the file name
    let subdir = pathParts.length > 1 ? pathParts[pathParts.length - 2] : 'unknown';
    imageDir = "results/" + subdir;
  }

  const apiURL = `https://api.github.com/repos/${owner}/${repo}/contents/${imageDir}`;
  const container = document.getElementById(containerId);
  if (!container) return;
  try {
    const res = await fetch(apiURL);
    if (!res.ok) throw new Error("Failed to fetch directory listing.");
    const files = await res.json();
    const imageFiles = files.filter(f => f.type === "file" && /\.(jpg|jpeg|png|gif|webp|svg)$/i.test(f.name));
    container.innerHTML = "";
    imageFiles.forEach(f => {
      const img = document.createElement("img");
      img.src = `https://${owner}.github.io/${imageDir}/${encodeURIComponent(f.name)}`;
      img.alt = f.name;
      img.style.maxWidth = "200px";
      img.style.margin = "5px";
      img.style.borderRadius = "5px";
      container.appendChild(img);
    });
    if (imageFiles.length === 0) {
      container.textContent = "No images found in /" + imageDir + ".";
    }
  } catch (e) {
    container.textContent = "Failed to load images.";
  }
}