let lastClickPosition = { x: 0, y: 0 };

// Listen for messages from content.js with click position
chrome.runtime.onMessage.addListener((message, sender) => {
  if (message.action === "storeClickPosition") {
    lastClickPosition = { x: message.x, y: message.y };
  }
});

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "detectMedia",
    title: "Detect Deepfake",
    contexts: ["all"]  // Works for image, video, audio, etc.
  });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  chrome.tabs.sendMessage(tab.id, {
    action: "detectMedia",
    url: info.srcUrl, // May be undefined for audio
    x: lastClickPosition.x,
    y: lastClickPosition.y
  });
});
