// PlayJev demo, English and Chinese copy. The page is authored in English (index.html) and this file carries the
// Chinese version plus the strings demo.js writes at runtime, in both languages.
//
// What is never translated: the move names and the prompt. They are the model's own input, so they stay exactly as
// the model saw them.
window.PJ_I18N = {
  en: {
    'ui.lang': '中文', 'ui.langTitle': '切换到中文',
    'ui.dark': 'dark', 'ui.light': 'light',
    'ui.darkTitle': 'switch to dark', 'ui.lightTitle': 'switch to light',
    'ui.score': 'score', 'ui.step': 'step', 'ui.speed': 'speed',
    'ui.loading': 'loading', 'ui.paused': 'paused',
    'ui.play': 'play', 'ui.pause': 'pause', 'ui.playall': 'play all', 'ui.pauseall': 'pause all',
    'ui.restart': 'restart', 'ui.next': 'next', 'ui.nextTitle': 'next recorded episode',
    'ui.trace': 'confidence through the episode', 'ui.conf': 'confidence',
    'ui.live': 'live server', 'ui.norec': 'no recording yet', 'ui.recording': 'recording',
    'ui.pending': 'pending', 'ui.mismatch': 'replayed score differs from the recording',
    'ui.pageerror': 'page error, see console', 'ui.whichrec': 'which recording to replay',
    'ui.gameframe': 'game frame',
    'ui.liveMode': 'Live mode: frames go to {url} and its probabilities pick every move.',
    'ui.noRecs': 'No recordings have been built into this page yet; the games load and wait.',
    'ui.duelAlone': 'The model alone', 'ui.duelS2': 'With the teacher',
    'ui.axP': 'p of the move it chose', 'ui.axModel': 'model', 'ui.axTeacher': 'teacher',
    'ui.axAlone': 'alone', 'ui.axFrames': 'frames of this game',
    'ui.byConf': 'the teacher takes the steps the model is least sure about',
    'ui.atRandom': 'the teacher takes the same share of steps at random',
    'ui.fromEight': 'fine-tuned from the eight-game model', 'ui.fromBase': 'fine-tuned from the raw base model',
    'ui.built': 'built',
  },
  zh: {
    'ui.lang': 'EN', 'ui.langTitle': 'switch to English',
    'ui.dark': '深色', 'ui.light': '浅色',
    'ui.darkTitle': '切换到深色', 'ui.lightTitle': '切换到浅色',
    'ui.score': '得分', 'ui.step': '步数', 'ui.speed': '速度',
    'ui.loading': '加载中', 'ui.paused': '已暂停',
    'ui.play': '播放', 'ui.pause': '暂停', 'ui.playall': '全部播放', 'ui.pauseall': '全部暂停',
    'ui.restart': '重播', 'ui.next': '下一局', 'ui.nextTitle': '换一局录像',
    'ui.trace': '整局的置信度', 'ui.conf': '置信度',
    'ui.live': '实时服务', 'ui.norec': '暂无录像', 'ui.recording': '录像',
    'ui.pending': '待补', 'ui.mismatch': '回放得分与录像不一致',
    'ui.pageerror': '页面报错，见控制台', 'ui.whichrec': '选择回放的录像',
    'ui.gameframe': '游戏画面',
    'ui.liveMode': '实时模式：画面发往 {url}，每一步都由它返回的概率决定。',
    'ui.noRecs': '这个页面还没有打包任何录像，游戏会加载后等待。',
    'ui.duelAlone': '模型独自玩', 'ui.duelS2': '背后有老师',
    'ui.axP': '所选动作的概率', 'ui.axModel': '模型', 'ui.axTeacher': '老师',
    'ui.axAlone': '独自', 'ui.axFrames': '该游戏的帧数',
    'ui.byConf': '老师接手模型最没把握的那些步', 'ui.atRandom': '老师随机接手同样比例的步',
    'ui.fromEight': '从八款游戏的模型继续微调', 'ui.fromBase': '从原始 base 模型微调',
    'ui.built': '构建于',

    'nav.gallery': '游戏墙', 'nav.results': '结果', 'nav.confidence': '置信度', 'nav.how': '原理',

    'lead.h1': '一个开源 0.8B 模型，只看画面就能玩十款浏览器游戏。',
    'lead.p': '一帧进去，一次前向，一个动作出来，单卡 43 毫秒。模型看到的只有画面，拿不到游戏内部状态，也从来不知道自己在玩哪一款。',
    'chip.params': '参数', 'chip.games': '款游戏', 'chip.move': '每步耗时',
    'chip.frames': '训练帧', 'chip.teacher': '相对老师',
    'link.code': '代码', 'link.weights': '权重', 'link.server': 'System One 服务',
    'link.reading': '论文清单', 'link.paper': '论文', 'link.soon': '即将发布',

    'gallery.h2': '游戏墙',
    'gallery.p': '十款游戏，一套权重，十套不同的可选动作。',
    'who.model': '模型', 'who.s2': '背后有老师', 'who.random': '随机策略',
    'mode.all': '十个一起跑', 'mode.solo': '一次跑一个',
    'btn.pauseall': '全部暂停',

    'results.h2': '结果',
    'results.p': '每款游戏 16 局留出种子，所有玩家用同一批种子，取概率最大的动作，不采样。条形表示模型落在随机策略与出标注的老师之间的什么位置。',
    'results.table': '同样的数字，表格版',
    'th.game': '游戏', 'th.random': '随机', 'th.model': '模型', 'th.teacher': '老师', 'th.vs': '相对老师',
    'th.from': '起点', 'th.frames': '帧数', 'th.agree': '一致率', 'th.score': '得分',

    'calib.h2': '它知道自己什么时候是对的吗？',
    'calib.p': '每个点取出模型给所选动作的概率大致相同的那些步，再看这些步里有多少次跟老师一致。落在对角线上，说明它报出的概率就是它真实的正确率。',

    'help.h2': '知道什么时候求助',
    'help.p': '有把握的步自己走，其余交给老师。同一款游戏、同一个种子的两块板：左边模型独自玩，右边它只保留有把握的步，其余由老师接手，接手的步标成珊瑚色。',
    'help.h3': '这个置信度值多少',
    'help.p2': '横轴是老师实际接手的步数比例，最左端是模型独自玩，最右端是老师全程接管。<b class="k-blue">按置信度挑出这些步</b>必须赢过<b class="k-coral">随机交出同样比例的步</b>，否则这个置信度就没有意义。',

    'transfer.h2': '学一款新游戏',
    'transfer.p': '从八款游戏的模型里留出两款。拿这个模型在留出游戏的几千帧上继续微调，与拿原始 base 模型在同样的帧上微调作对比。曲线是微调后的模型在这款它没训练过的游戏的 2000 帧上，有多少次选中了老师的动作。',
    'transfer.table': '每一次运行的表格，含得分',

    'how.h2': '原理',
    'how.p1': '模型拿到画面和可选动作列表，按冻结的 OpenJev 提示词渲染，答案就是 "Answer:" 之后的那一个 token。动作上的概率来自该位置选项字母上的 softmax：一次前向，不采样，不生成任何文本。',
    'how.p2': '执行概率最大的那个动作，只有一个例外：如果某个动作不会让画面发生变化，比如 2048 或推箱子里被挡住的方向，就不会在同一张画面上重复它，改走概率次大的那个。',
    'how.p3': '这里每一块板都是训练时用的那份游戏代码，配同样的 shim 和 hook：虚拟时钟、带种子的随机源、合成按键事件，所以录下的动作序列能一步不差地重放出整局。',
    'how.optionlist': '可选动作列表',

    'credits.h2': '十款游戏',
    'credits.p': '十款开源浏览器游戏，各自带着原作者的许可证一起收录，我们改过的每一行都记录在旁边。超级马里奥、Floppy Bird 和赛车的美术素材版权属于各自的原始持有者，下面的许可证只覆盖作者自己写的代码。',
  },
  // Game names for the Chinese page. The move names stay English: they are the model's input.
  games: {
    tetris: '俄罗斯方块', snake: '贪吃蛇', pacman: '吃豆人', racer: '赛车', invaders: '太空侵略者',
    sokoban: '推箱子', mario: '超级马里奥', flappy: '飞翔的小鸟', breakout: '打砖块', '2048': '2048',
  },
};
