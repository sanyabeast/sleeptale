# YouTube Music Library Download Helper

## Overview

This is a Tampermonkey userscript that enhances the YouTube Music Library interface to simplify the process of downloading tracks with consistent, formatted filenames for use with DeepVideo2.

## Purpose

When working with the DeepVideo2 project, having consistently named music files is essential for proper categorization and selection. This script automatically formats music track information from YouTube's Audio Library into the specific naming convention required by DeepVideo2:

```
NAME_track_title_GENRE_genre_MOOD_mood_ARTIST_artist_name
```

## Installation

1. Install the [Tampermonkey](https://www.tampermonkey.net/) browser extension for your browser:
   - [Chrome](https://chrome.google.com/webstore/detail/tampermonkey/dhdgffkkebhmkfjojejmpbldmpobfkfo)
   - [Firefox](https://addons.mozilla.org/en-US/firefox/addon/tampermonkey/)
   - [Edge](https://microsoftedge.microsoft.com/addons/detail/tampermonkey/iikmkjmpaadaobahmlepeloendndfphd)

2. Create a new script in Tampermonkey and paste the contents of `dl_helper.js` into it.

3. Save the script and ensure it's enabled.

## Usage

1. Navigate to the YouTube Audio Library at https://studio.youtube.com/channel/*/music

2. Click on any track in the library.

3. The script will automatically copy a formatted filename to your clipboard containing:
   - The track name (NAME)
   - The genre (GENRE)
   - The mood (MOOD)
   - The artist name (ARTIST)

4. When downloading the track, paste this formatted name as the filename.

5. Place the downloaded files in your DeepVideo2 `lib/music` directory.

## How It Works

The script:
1. Listens for click events on track rows in the YouTube Music Library
2. Extracts track metadata (title, genre, mood, artist)
3. Sanitizes the text and converts it to snake_case (except for artist names, which preserve spaces)
4. Formats the information into the DeepVideo2 naming convention
5. Copies the formatted filename to your clipboard

## Format Details

The script produces filenames in this format:
```
NAME_track_title_GENRE_genre_MOOD_mood_ARTIST_artist name
```

Note that:
- Track title, genre, and mood are converted to snake_case (lowercase with underscores)
- Artist names preserve their original spacing (not converted to snake_case)
- All punctuation is removed
- All text is converted to lowercase

### Example
```
Original: "Chosen One, Hip-Hop/Rap, Dark, Verified Picasso"
Formatted: "NAME_chosen_one_GENRE_hiphop_rap_MOOD_dark_ARTIST_verified picasso"
```

## Customization

If you need to modify the formatting pattern, edit the `formatted` variable in the `handleTrackClick` function:

```javascript
const formatted = `NAME_${name}_GENRE_${genre}_MOOD_${mood}_ARTIST_${artist}`;
```

## Troubleshooting

- If the script doesn't work after installation, refresh the YouTube Music Library page.
- Make sure Tampermonkey is enabled for the YouTube Studio domain.
- Check your browser's console for any error messages (press F12 to open developer tools).

## Integration with DeepVideo2

This script is designed to work seamlessly with the DeepVideo2 project, which uses music files with specific naming conventions to enable better selection based on genre and mood during video generation.

After downloading music files with properly formatted names, you can use the `name_music.py` script in the DeepVideo2 root directory to ensure all music files follow the same naming convention.
