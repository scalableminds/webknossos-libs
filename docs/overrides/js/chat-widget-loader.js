(function () {
  // A single assistant answers questions about both the WEBKNOSSOS application
  // and the Python library, API and CLI, so it is shown on every docs page.
  import("https://cdn.n8nchatui.com/v1/embed.js").then(function (module) {
    module.default.init({
      n8nChatUrl: "https://chat.webknossos.org/chat",
      metadata: {},
      theme: {
        button: {
          backgroundColor: "#5660ff",
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
          tooltipMessage: "Hello 👋 I am here to help you with WEBKNOSSOS and its Python library",
          tooltipBackgroundColor: "#5660ff",
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
          title: "WEBKNOSSOS Assistant",
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
            backgroundColor: "#5660ff",
            textColor: "#fafafa",
            showAvatar: false,
            avatarSrc: "https://static.webknossos.org/logos/webknossos-icon-only.svg",
            showCopyToClipboardIcon: false
          },
          userMessage: {
            backgroundColor: "#a8b4ff",
            textColor: "#050505",
            showAvatar: false,
            avatarSrc: "https://www.svgrepo.com/show/532363/user-alt-1.svg"
          },
          textInput: {
            placeholder: "Type your query",
            backgroundColor: "#ffffff",
            textColor: "#1e1e1f",
            sendButtonColor: "#5660ff",
            maxChars: 500,
            maxCharsWarningMessage: "You exceeded the characters limit. Please input less than 50 characters.",
            autoFocus: false,
            borderRadius: 6,
            sendButtonBorderRadius: 50
          }
        }
      }
    });
  });
})();
