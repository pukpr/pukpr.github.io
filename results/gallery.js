/**
 * Dynamically loads images from a GitHub repo directory into a container.
 * @param {Object} opts
 * @param {string} opts.owner - GitHub username/organization
 * @param {string} opts.repo - GitHub repository name
 * @param {string} opts.imageDir - Directory inside the repo (e.g. 'images')
 * @param {string} opts.containerId - ID of the DOM element to render images into
 */
async function loadGithubGallery({ owner, repo, imageDir, containerId }) {
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