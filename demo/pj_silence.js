// The roster ships silent: every sound file was deleted from the vendored games. The game code
// still asks for them, so on a served page each of those requests is a 404 in the console. This
// cuts the request before it is made. Demo only. During collection the driver aborts audio
// requests instead (playjev/env.py), and the shim already keeps every media element muted with a
// play() that resolves and does nothing, so no page in this project has ever produced a sound.
(function () {
  const SRC = new WeakMap();
  const AUDIO = /\.(mp3|ogg|wav|m4a|mid|midi|oga|flac)(\?|#|$)/i;
  const keep = (el, v) => { SRC.set(el, v == null ? '' : String(v)); };

  try {
    const media = HTMLMediaElement.prototype;
    Object.defineProperty(media, 'src', {
      configurable: true,
      get() { return SRC.get(this) || ''; },
      set(v) { keep(this, v); },              // remembered, never fetched
    });
    media.load = function () {};
    media.play = function () { return Promise.resolve(); };
    media.pause = function () {};
  } catch (e) {}

  // new Audio(url) sets the content attribute inside the constructor, past the property above
  try {
    window.Audio = function (url) {
      const el = document.createElement('audio');
      if (url != null) keep(el, url);
      return el;
    };
  } catch (e) {}

  // buzz (flappy) builds a <source> child and assigns its src property
  try {
    const source = HTMLSourceElement.prototype;
    const real = Object.getOwnPropertyDescriptor(source, 'src');
    Object.defineProperty(source, 'src', {
      configurable: true,
      get() { return SRC.has(this) ? SRC.get(this) : real.get.call(this); },
      set(v) { AUDIO.test(String(v)) ? keep(this, v) : real.set.call(this, v); },
    });
  } catch (e) {}

  // soundmanager (breakout) sets the attribute instead
  const setAttribute = Element.prototype.setAttribute;
  Element.prototype.setAttribute = function (name, value) {
    const n = String(name).toLowerCase();
    const media = this instanceof HTMLMediaElement;
    if (n === 'src' && (media || (this instanceof HTMLSourceElement && AUDIO.test(String(value))))) {
      return keep(this, value);
    }
    return setAttribute.call(this, name, value);
  };
})();
