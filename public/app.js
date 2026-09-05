(() => {
  const API_BASE =
    window.location.hostname === 'localhost' ||
    window.location.hostname === '127.0.0.1'
    ? 'http://127.0.0.1:8000'
    : '';

  const screens = {
    home: document.querySelector('#home'),
    game: document.querySelector('#game'),
    result: document.querySelector('#result')
  };

  const els = {
    play: document.querySelector('#playBtn'),
    replay: document.querySelector('#replayBtn'),
    quit: document.querySelector('#quitBtn'),

    form: document.querySelector('#guessForm'),
    input: document.querySelector('#guessInput'),

    timer: document.querySelector('#timer'),
    attempts: document.querySelector('#attempts'),

    heatNumber: document.querySelector('#heatNumber'),
    heatLabel: document.querySelector('#heatLabel'),
    meterFill: document.querySelector('#meterFill'),

    history: document.querySelector('#history'),
    bestLabel: document.querySelector('#bestLabel'),

    error: document.querySelector('#error'),

    nearestWords: document.querySelector('#nearestWords'),
    nearestContext: document.querySelector('#nearestContext'),

    hintBtn: document.querySelector('#hintBtn'),
    hintList: document.querySelector('#hintList'),
    hintCount: document.querySelector('#hintCount'),

    answer: document.querySelector('#answer'),
    resultEyebrow: document.querySelector('#resultEyebrow'),
    resultTitle: document.querySelector('#result-title'),

    finalTime: document.querySelector('#finalTime'),
    finalAttempts: document.querySelector('#finalAttempts'),
    finalBest: document.querySelector('#finalBest'),

    trail: document.querySelector('#trail')
  };

  let gameId = null;
  let guesses = [];
  let hints = [];
  let best = 0;

  let startTime = 0;
  let timerId = null;

  let finished = false;


  // --------------------------------------------------
  // SCREEN MANAGEMENT
  // --------------------------------------------------

  function showScreen(name) {
    Object.values(screens).forEach(screen => {
      screen.classList.remove('active');
    });

    if (screens[name]) {
      screens[name].classList.add('active');
    }
  }


  // --------------------------------------------------
  // TIME
  // --------------------------------------------------

  function formatTime(milliseconds) {
    const totalSeconds = Math.max(
      0,
      Math.floor(milliseconds / 1000)
    );

    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;

    return (
      String(minutes).padStart(2, '0') +
      ':' +
      String(seconds).padStart(2, '0')
    );
  }


  // --------------------------------------------------
  // SCORE LABEL
  // --------------------------------------------------

  function getHeatLabel(score) {
    score = Number(score) || 0;

    if (score < 20) return 'Ice cold';
    if (score < 35) return 'Cold';
    if (score < 50) return 'Getting warmer';
    if (score < 65) return 'Warm';
    if (score < 80) return 'Hot';
    if (score < 95) return 'Very hot';

    return 'Almost there';
  }


  // --------------------------------------------------
  // RESET GAME UI
  // --------------------------------------------------

  function resetUI() {
    clearInterval(timerId);

    els.timer.textContent = '00:00';
    els.attempts.textContent = '0';

    els.heatNumber.textContent = '—';
    els.heatLabel.textContent = 'Make your first guess';

    els.meterFill.style.width = '0%';

    els.bestLabel.textContent = 'Best —';

    els.history.innerHTML = '';

    els.nearestWords.innerHTML =
      '<li class="nearest-empty">' +
      'Nearest words will appear here after your first guess.' +
      '</li>';

    els.nearestContext.textContent = 'Waiting for a guess';

    els.hintList.innerHTML =
      '<li class="hint-empty">' +
      'Use a hint when you get stuck.' +
      '</li>';

    els.hintCount.textContent = '0 / 4';

    els.error.textContent = '';

    els.input.value = '';

    // Reset End Game
    els.quit.disabled = false;
    els.quit.textContent = 'End game';

    // Reset Hint
    els.hintBtn.disabled = false;
    els.hintBtn.textContent = 'Hint ?';
  }


  // --------------------------------------------------
  // API HELPER
  // --------------------------------------------------

  async function api(path, options = {}) {
    let response;

    try {
      response = await fetch(
        `${API_BASE}${path}`,
        {
          ...options,
          headers: {
            'Content-Type': 'application/json',
            ...(options.headers || {})
          }
        }
      );
    } catch (error) {
      throw new Error(
        'Cannot connect to WordHeat API. ' +
        'Make sure the backend is running on port 8000.'
      );
    }

    let data = {};

    try {
      data = await response.json();
    } catch (_) {
      data = {};
    }

    if (!response.ok) {
      throw new Error(
        data.detail ||
        `Request failed (${response.status})`
      );
    }

    return data;
  }


  // --------------------------------------------------
  // START GAME
  // --------------------------------------------------

  async function startGame() {
    clearInterval(timerId);

    gameId = null;
    finished = false;

    guesses = [];
    hints = [];
    best = 0;

    resetUI();

    try {
      const data = await api(
        '/api/game/start',
        {
          method: 'POST',
          body: JSON.stringify({})
        }
      );

      gameId = data.game_id;

      if (!gameId) {
        throw new Error(
          'Backend did not return a game ID.'
        );
      }

      startTime = Date.now();

      timerId = setInterval(() => {
        if (!finished && gameId) {
          els.timer.textContent =
            formatTime(Date.now() - startTime);
        }
      }, 250);

      showScreen('game');

      setTimeout(() => {
        els.input.focus();
      }, 50);

    } catch (error) {
      showScreen('home');

      alert(
        'Could not start WordHeat.\n\n' +
        error.message +
        '\n\n' +
        'Make sure the backend is running and vectors are prepared.'
      );
    }
  }


  // --------------------------------------------------
  // RENDER GUESS HISTORY
  // --------------------------------------------------

  function renderHistory(history) {
    els.history.innerHTML = '';

    history
      .slice()
      .reverse()
      .forEach((guess, reverseIndex) => {

        const li = document.createElement('li');

        const index = document.createElement('span');
        const word = document.createElement('span');
        const score = document.createElement('span');

        index.className = 'index';
        score.className = 'score';

        index.textContent =
          String(history.length - reverseIndex)
            .padStart(2, '0');

        word.textContent = guess.word;

        score.textContent =
          `${guess.score}%`;

        li.append(
          index,
          word,
          score
        );

        els.history.appendChild(li);
      });
  }


  // --------------------------------------------------
  // RENDER NEAREST WORDS
  // --------------------------------------------------

  function renderNearest(input, nearest) {
    els.nearestWords.innerHTML = '';

    els.nearestContext.textContent =
      `Near "${input}"`;

    if (
      !nearest ||
      nearest.length === 0
    ) {
      els.nearestWords.innerHTML =
        '<li class="nearest-empty">' +
        'No nearby words found.' +
        '</li>';

      return;
    }

    nearest.forEach((item, index) => {

      const li = document.createElement('li');

      const rank = document.createElement('span');
      const word = document.createElement('span');
      const score = document.createElement('span');

      rank.className = 'rank';
      word.className = 'word';
      score.className = 'score';

      rank.textContent =
        String(index + 1).padStart(2, '0');

      word.textContent = item.word;

      score.textContent =
        `${item.score}%`;

      li.append(
        rank,
        word,
        score
      );

      els.nearestWords.appendChild(li);
    });
  }


  // --------------------------------------------------
  // RENDER HINTS
  // --------------------------------------------------

  function renderHints() {
    els.hintCount.textContent =
      `${hints.length} / 4`;

    els.hintList.innerHTML = '';

    if (hints.length === 0) {
      els.hintList.innerHTML =
        '<li class="hint-empty">' +
        'Use a hint when you get stuck.' +
        '</li>';

      return;
    }

    hints.forEach((item, index) => {

      const li = document.createElement('li');

      const rank = document.createElement('span');
      rank.className = 'rank';

      rank.textContent =
        String(index + 1).padStart(2, '0');


      const word = document.createElement('span');
      word.className = 'hint-word';

      word.textContent = item.word;


      const score = document.createElement('span');
      score.className = 'hint-score';

      score.textContent =
        `${item.score}%`;


      li.append(
        rank,
        word,
        score
      );

      els.hintList.appendChild(li);
    });


    // Disable after 4 hints
    if (hints.length >= 4) {

      els.hintBtn.disabled = true;
      els.hintBtn.textContent = 'No hints left';

    } else {

      els.hintBtn.disabled = false;
      els.hintBtn.textContent = 'Hint ?';
    }
  }


  // --------------------------------------------------
  // FINISH GAME
  // --------------------------------------------------

  function finishGame(
    answer,
    history,
    endedByPlayer = false
  ) {

    finished = true;

    clearInterval(timerId);

    gameId = null;

    els.answer.textContent =
      String(answer).toUpperCase();

    els.finalTime.textContent =
      formatTime(
        Date.now() - startTime
      );

    els.finalAttempts.textContent =
      String(history.length);

    els.finalBest.textContent =
      `${best}%`;


    // Result screen text
    if (endedByPlayer) {

      els.resultEyebrow.textContent =
        'Game ended';

      els.resultTitle.textContent =
        'The hidden word was';

    } else {

      els.resultEyebrow.textContent =
        'Game complete';

      els.resultTitle.textContent =
        'You found it.';
    }


    // Semantic trail
    els.trail.innerHTML = '';

    history.forEach((guess, index) => {

      const span =
        document.createElement('span');

      span.textContent =
        `${guess.word} · ${guess.score}%`;

      els.trail.appendChild(span);


      if (index < history.length - 1) {

        const arrow =
          document.createElement('b');

        arrow.textContent = '→';

        els.trail.appendChild(arrow);
      }
    });


    // Add answer
    const answerSpan =
      document.createElement('span');

    answerSpan.textContent =
      String(answer);

    answerSpan.style.fontWeight = '600';

    els.trail.appendChild(answerSpan);


    showScreen('result');
  }


  // --------------------------------------------------
  // SUBMIT GUESS
  // --------------------------------------------------

  async function submitGuess(event) {

    event.preventDefault();

    if (finished || !gameId) {
      return;
    }

    const value =
      els.input.value
        .trim()
        .toLowerCase();

    els.error.textContent = '';

    if (!value) {
      return;
    }


    const button =
      els.form.querySelector('.primary-btn');

    const oldText =
      button.innerHTML;

    button.disabled = true;
    button.textContent = 'Checking…';


    try {

      const data =
        await api(
          '/api/game/guess',
          {
            method: 'POST',
            body: JSON.stringify({
              game_id: gameId,
              word: value
            })
          }
        );


      guesses =
        data.history || [];


      best =
        Math.max(
          best,
          Number(data.score) || 0
        );


      // Update score
      els.attempts.textContent =
        String(guesses.length);

      els.heatNumber.textContent =
        `${data.score}%`;

      els.heatLabel.textContent =
        getHeatLabel(data.score);

      els.meterFill.style.width =
        `${data.score}%`;

      els.bestLabel.textContent =
        `Best ${best}%`;


      // Update panels
      renderHistory(guesses);

      renderNearest(
        value,
        data.nearest_words
      );


      els.input.value = '';

      els.input.focus();


      // WIN
      if (data.won) {

        setTimeout(() => {

          finishGame(
            value,
            guesses,
            false
          );

        }, 300);
      }

    } catch (error) {

      els.error.textContent =
        error.message;

      els.input.focus();

    } finally {

      button.disabled = false;
      button.innerHTML = oldText;
    }
  }


  // --------------------------------------------------
  // REQUEST HINT
  // --------------------------------------------------

  async function requestHint() {

    if (finished || !gameId) {
      return;
    }

    els.error.textContent = '';


    if (hints.length >= 4) {

      els.error.textContent =
        'All 4 hints have already been used.';

      return;
    }


    const oldText =
      els.hintBtn.textContent;

    els.hintBtn.disabled = true;
    els.hintBtn.textContent =
      'Finding…';


    try {

      const data =
        await api(
          '/api/game/hint',
          {
            method: 'POST',
            body: JSON.stringify({
              game_id: gameId
            })
          }
        );


      if (
        !data.hint ||
        !data.hint.word
      ) {

        throw new Error(
          'No hint was returned by the backend.'
        );
      }


      // Store complete hint object
      hints.push(data.hint);


      // Render it
      renderHints();


      // Focus input
      els.input.focus();

    } catch (error) {

      els.error.textContent =
        error.message;

      els.hintBtn.disabled = false;
      els.hintBtn.textContent =
        oldText;
    }
  }


  // --------------------------------------------------
  // END GAME
  // --------------------------------------------------

  async function endGame() {

    if (finished || !gameId) {
      return;
    }


    const endingGameId =
      gameId;


    // Immediately disable controls
    els.quit.disabled = true;
    els.hintBtn.disabled = true;

    els.quit.textContent =
      'Ending…';


    try {

      // IMPORTANT:
      // No confirm()
      // No permission dialog
      // Directly call backend.
      const data =
        await api(
          '/api/game/end',
          {
            method: 'POST',
            body: JSON.stringify({
              game_id: endingGameId
            })
          }
        );


      // Backend directly returns answer
      finishGame(
        data.answer,
        data.history || guesses,
        true
      );

    } catch (error) {

      els.error.textContent =
        error.message;

      els.quit.disabled = false;

      els.hintBtn.disabled =
        hints.length >= 4;

      els.quit.textContent =
        'End game';
    }
  }


  // --------------------------------------------------
  // BUTTON EVENT LISTENERS
  // --------------------------------------------------

  if (els.play) {
    els.play.addEventListener(
      'click',
      startGame
    );
  }


  if (els.replay) {
    els.replay.addEventListener(
      'click',
      startGame
    );
  }


  if (els.form) {
    els.form.addEventListener(
      'submit',
      submitGuess
    );
  }


  if (els.hintBtn) {
    els.hintBtn.addEventListener(
      'click',
      requestHint
    );
  }


  if (els.quit) {
    els.quit.addEventListener(
      'click',
      endGame
    );
  }

})();