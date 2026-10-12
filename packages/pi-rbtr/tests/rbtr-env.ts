/**
 * The rbtr settings every test process and daemon uses: no embedding and no
 * reranker, so no model runs. Every extension feature works without them.
 */
export const TEST_RBTR_ENV = {
  RBTR_EMBED: "false",
  RBTR_RERANKER_MODEL: "",
};
