from pathlib import Path

APP = Path(__file__).resolve().parent / "app.py"
OLD = "    st.markdown(\"<div class='ow-up-grid'>\" + \"\".join(cards) + \"</div>\", unsafe_allow_html=True)"
NEW = '''    full_html = "<div class='ow-up-grid'>" + "".join(cards) + "</div>"
    if hasattr(st, "html"):
        st.html(full_html)
    else:
        # Compact the generated cards so Markdown cannot treat indentation inside
        # sibling HTML blocks as a four-space code block on mobile clients.
        st.markdown("".join(line.strip() for line in full_html.splitlines()), unsafe_allow_html=True)'''

text = APP.read_text(encoding="utf-8")
if OLD in text:
    text = text.replace(OLD, NEW, 1)
    APP.write_text(text, encoding="utf-8")
    print("Applied Batter Upside mobile HTML render fix.")
elif "full_html = \"<div class='ow-up-grid'>\" + \"\".join(cards) + \"</div>\"" in text:
    print("Batter Upside mobile HTML render fix already present.")
else:
    print("Batter Upside renderer signature changed; startup patch skipped safely.")
