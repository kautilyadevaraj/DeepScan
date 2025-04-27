// Track last right-click position
document.addEventListener("mousedown", (e) => {
  if (e.button === 2) {
    chrome.runtime.sendMessage({
      action: "storeClickPosition",
      x: e.clientX,
      y: e.clientY,
    });
  }
});

chrome.runtime.onMessage.addListener(async (message) => {
  if (message.action === "detectMedia") {
    const mediaType = detectElementTypeAtClick(message.x, message.y);

    if (mediaType === "image") {
      const imageElement = getElementAtClick(message.x, message.y);
      const imageUrl = imageElement?.src;

      if (!imageUrl) return alert("⚠️ Couldn't fetch image source.");

      // Highlight the image with a border
      imageElement.style.border = "5px solid red";

      // Add loading spinner on the image element (No spinning animation)
      const spinner = document.createElement("div");
      spinner.style.position = "absolute";  // Absolutely position it on top of the image
      spinner.style.top = "50%";
      spinner.style.left = "50%";
      spinner.style.transform = "translate(-50%, -50%)"; // Centering the spinner on the media element
      spinner.style.border = "8px solid #f3f3f3";
      spinner.style.borderTop = "8px solid #3498db";
      spinner.style.borderRadius = "50%";
      spinner.style.width = "50px";
      spinner.style.height = "50px";
      spinner.style.zIndex = 10000; // Ensure it's on top of the media element
      spinner.style.pointerEvents = "none"; // So it won't block interaction with the element

      // Ensure the image is positioned relative so the spinner is positioned correctly within it
      imageElement.style.position = "relative"; 
      imageElement.appendChild(spinner);  // Attach the spinner directly to the image

      await ensureSandbox();

      const resp = await fetch(imageUrl);
      const buffer = await resp.arrayBuffer();

      const id = crypto.randomUUID();
      const promise = new Promise((res) => (pending[id] = res));
      sandboxFrame.contentWindow.postMessage({ id, imageBytes: buffer }, "*");

      const result = await promise;

      // Remove loading spinner after receiving result
      imageElement.removeChild(spinner);

      // Remove the border highlight
      imageElement.style.border = "";

      // 🚀 Send to backend for detection
      try {
        const response = await fetch("http://127.0.0.1:5000/predict", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ image: result.base64 }),
        });

        const detection = await response.json();

        const { prediction, probabilities } = detection;

        if (prediction === "deepfake") {
          alert(
            `⚠️ Deepfake Detected!\nProbabilities: ${JSON.stringify(
              probabilities
            )}`
          );
        } else if (prediction === "ai_gen") {
          alert(
            `🤖 AI-Generated Content Detected!\nProbabilities: ${JSON.stringify(
              probabilities
            )}`
          );
        } else if (prediction === "real") {
          alert(
            `✅ Looks Real!\nProbabilities: ${JSON.stringify(probabilities)}`
          );
        } else {
          alert(
            `❓ Unknown Prediction: ${prediction}\nProbabilities: ${JSON.stringify(
              probabilities
            )}`
          );
        }
      } catch (err) {
        alert("❌ Failed to connect to detection server.");
        console.error(err);
      }
    } else if (mediaType === "video") {
      alert("🎥 Detected a Video!");
    } else if (mediaType === "audio") {
      alert("🎧 Detected an Audio!");
    } else {
      alert("❓ Unsupported media or not detected.");
    }
  }
});

// Remove spinner animation (No spinning effect)
const style = document.createElement('style');
style.innerHTML = `
  /* Removed spinning animation, just a static loader */
`;
document.head.appendChild(style);

function detectElementTypeAtClick(x, y) {
  let el = document.elementFromPoint(x, y);
  if (!el) return null;

  let tag = el.tagName?.toLowerCase();
  if (tag === "img") return "image";
  if (tag === "video") return "video";
  if (tag === "audio") return "audio";

  const mediaChild = el.querySelector?.("img, video, audio");
  if (mediaChild) {
    tag = mediaChild.tagName.toLowerCase();
    if (tag === "img") return "image";
    if (tag === "video") return "video";
    if (tag === "audio") return "audio";
  }

  while (el && el.tagName) {
    tag = el.tagName.toLowerCase();
    if (tag === "img") return "image";
    if (tag === "video") return "video";
    if (tag === "audio") return "audio";
    el = el.parentElement;
  }

  return null;
}

function getElementAtClick(x, y) {
  let el = document.elementFromPoint(x, y);
  if (!el) return null;

  if (el.tagName.toLowerCase() === "img") return el;

  const imgChild = el.querySelector?.("img");
  if (imgChild) return imgChild;

  while (el && el.tagName) {
    if (el.tagName.toLowerCase() === "img") return el;
    el = el.parentElement;
  }

  return null;
}

let sandboxFrame;
let pending = {};

async function ensureSandbox() {
  if (sandboxFrame) return;
  sandboxFrame = document.createElement("iframe");
  sandboxFrame.src = chrome.runtime.getURL("sandbox.html");
  sandboxFrame.style.display = "none";
  document.documentElement.appendChild(sandboxFrame);
  await new Promise((res) => (sandboxFrame.onload = res));

  window.addEventListener("message", (evt) => {
    const { id, result } = evt.data;
    if (pending[id]) {
      pending[id](result);
      delete pending[id];
    }
  });
}
