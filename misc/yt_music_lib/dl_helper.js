// ==UserScript==
// @name         YouTube Music Library Download Helper
// @namespace    http://tampermonkey.net/
// @version      0.6
// @description  Copies cleaned track info to clipboard in snake_case for filename use
// @author       You
// @match        https://studio.youtube.com/channel/*/music
// @grant        GM.setClipboard
// @run-at       document-end
// ==/UserScript==

(function () {
    'use strict';
  
    // Sanitizes and converts to lowercase snake_case
    function sanitizeToSnakeCase(value) {
      return value
        .toLowerCase()
        .replace(/[^\w\s]/g, '')       // Remove punctuation
        .replace(/\s+/g, '_')          // Replace spaces with underscores
        .replace(/^_+|_+$/g, '');      // Trim leading/trailing underscores
    }
  
    function extractField(row, id) {
      const el = row.querySelector(`#${id}`);
      return sanitizeToSnakeCase(el?.textContent || "");
    }
  
    function extractArtist(row) {
      const artistEl = row.querySelector('#artist .text');
      return sanitizeToSnakeCase(artistEl?.textContent || "");
    }
  
    function handleTrackClick(rowElement) {
      const name = extractField(rowElement, "title");
      const genre = extractField(rowElement, "genre");
      const mood = extractField(rowElement, "mood");
      const artist = extractArtist(rowElement);
  
      const formatted = `NAME_${name}_GENRE_${genre}_MOOD_${mood}_ARTIST_${artist}`;
      GM.setClipboard(formatted, "text");
      console.log("📋 Copied to clipboard (for filename):", formatted);
    }
  
    document.addEventListener("click", function (event) {
      if (event.button !== 0) return; // Only left click
      const row = event.target.closest("ytmus-library-row");
      if (row) {
        handleTrackClick(row);
      }
    });
  
    console.log("✅ YouTube Music Library Filename Helper active.");
  })();
  