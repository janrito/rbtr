// Build configuration for the greeter.
export default defineConfig({
  root: "src",
});

declare module "greeter-env" {
  export const locale: string;
}

registerGreeter("en");
