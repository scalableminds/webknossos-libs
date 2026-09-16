(function () {
  // Each section of the docs has its own assistant (backed by a different
  // webhook) with its own color scheme. Only the widget matching the current
  // page is shown; pages outside of these prefixes show no widget at all.
  var WIDGETS = [
    {
      id: "webknossos",
      prefixes: ["/webknossos/"],
      n8nChatUrl: "https://docs.webknossos.org/webhooks/webknossos/ask",
      title: "WEBKNOSSOS Assistant",
      tooltipMessage: "Hello 👋 I am here to help you with WEBKNOSSOS",
      colors: {
        primary: "#5660ff",
        userMessage: "#a8b4ff",
      },
    },
    {
      id: "webknossos-py",
      prefixes: ["/webknossos-py/", "/api/", "/cli/"],
      n8nChatUrl: "https://docs.webknossos.org/webhooks/webknossos-py/ask",
      title: "WEBKNOSSOS Python Assistant",
      tooltipMessage: "Hello 👋 I am here to help you with the Python library, API and CLI",
      colors: {
        primary: "#00a79d",
        userMessage: "#9fe0da",
      },
    },
  ];

  // The embed script keeps a single popup instance: calling init() again
  // replaces the popup that is currently in the DOM. So switching sections
  // means re-initializing rather than toggling between two popups.
  var modulePromise = null;
  var currentId = null;
  var initSeq = 0;

  function matches(widget) {
    return widget.prefixes.some(function (prefix) {
      return window.location.pathname.startsWith(prefix);
    });
  }

  function activeWidget() {
    for (var i = 0; i < WIDGETS.length; i++) {
      if (matches(WIDGETS[i])) return WIDGETS[i];
    }
    return null;
  }

  function config(widget) {
    return {
      n8nChatUrl: widget.n8nChatUrl,
      metadata: {},
      theme: {
        button: {
          backgroundColor: widget.colors.primary,
          right: 20,
          bottom: 20,
          size: 50,
          iconColor: "#373434",
          customIconSrc: "https://www.svgrepo.com/show/362552/chat-centered-dots-bold.svg",
          customIconSize: 60,
          customIconBorderRadius: 15,
          autoWindowOpen: { autoOpen: false, openDelay: 2 },
          borderRadius: "rounded",
          draggable: false
        },
        tooltip: {
          showTooltip: true,
          tooltipMessage: widget.tooltipMessage,
          tooltipBackgroundColor: widget.colors.primary,
          tooltipTextColor: "#ffffff",
          tooltipFontSize: 15,
          hideTooltipOnMobile: true
        },
        allowProgrammaticMessage: false,
        chatWindow: {
          borderRadiusStyle: "rounded",
          avatarBorderRadius: 25,
          messageBorderRadius: 6,
          showTitle: true,
          title: widget.title,
          titleAvatarSrc: "https://www.svgrepo.com/show/362552/chat-centered-dots-bold.svg",
          avatarSize: 40,
          welcomeMessage: "Hello! How can I help you today?",
          errorMessage: "I lost connection to the mothership. Please email webknossos-support@scalableminds.com instead",
          backgroundColor: "#ffffff",
          height: 600,
          width: 400,
          fontSize: 16,
          starterPromptFontSize: 15,
          renderHTML: false,
          clearChatOnReload: false,
          showScrollbar: false,
          botMessage: {
            backgroundColor: widget.colors.primary,
            textColor: "#fafafa",
            showAvatar: false,
            avatarSrc: "https://static.webknossos.org/logos/webknossos-icon-only.svg",
            showCopyToClipboardIcon: false
          },
          userMessage: {
            backgroundColor: widget.colors.userMessage,
            textColor: "#050505",
            showAvatar: false,
            avatarSrc: "https://www.svgrepo.com/show/532363/user-alt-1.svg"
          },
          textInput: {
            placeholder: "Type your query",
            backgroundColor: "#ffffff",
            textColor: "#1e1e1f",
            sendButtonColor: widget.colors.primary,
            maxChars: 500,
            maxCharsWarningMessage: "You exceeded the characters limit. Please input less than 50 characters.",
            autoFocus: false,
            borderRadius: 6,
            sendButtonBorderRadius: 50
          }
        }
      }
    };
  }

  function setPopupVisible(visible) {
    document.querySelectorAll("n8nchatui-popup").forEach(function (el) {
      el.style.display = visible ? "" : "none";
    });
  }

  // init() may create the popup asynchronously, and may reuse the element we
  // just hid, so keep un-hiding for a moment after initializing.
  function reveal(token, attemptsLeft) {
    if (token !== initSeq) return;
    setPopupVisible(true);
    if (attemptsLeft > 0) {
      setTimeout(function () { reveal(token, attemptsLeft - 1); }, 100);
    }
  }

  function updateWidget() {
    var active = activeWidget();
    if (active == null) {
      // Leaving the documented sections: keep the instance, just hide it.
      initSeq++;
      setPopupVisible(false);
      return;
    }
    if (active.id === currentId) {
      setPopupVisible(true);
      return;
    }
    currentId = active.id;
    var token = ++initSeq;
    setPopupVisible(false);
    if (modulePromise == null) {
      modulePromise = import("https://cdn.n8nchatui.com/v1/embed.js");
    }
    modulePromise.then(function (module) {
      // A newer navigation happened while the module was loading.
      if (token !== initSeq) return;
      module.default.init(config(active));
      reveal(token, 20);
    });
  }

  // Intercept history.pushState and history.replaceState for SPA navigation
  function wrap(method) {
    var original = history[method];
    history[method] = function () {
      original.apply(this, arguments);
      updateWidget();
    };
  }
  wrap('pushState');
  wrap('replaceState');

  // Browser back/forward navigation
  window.addEventListener('popstate', updateWidget);

  // Initial page load
  updateWidget();
})();
