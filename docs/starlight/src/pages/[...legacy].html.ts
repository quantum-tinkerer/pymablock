import type { APIRoute, GetStaticPaths } from 'astro';
import { sources } from '../../navigation.mjs';
import { base } from '../../deployment.mjs';

// Retain the original Sphinx page URLs as static Astro redirects.
export const getStaticPaths: GetStaticPaths = () => Object.values(sources)
  .filter(({ id }) => id !== 'index')
  .map(({ id }) => ({ params: { legacy: id }, props: { id } }));

export const GET: APIRoute = ({ props, redirect }) =>
  redirect(`${base.replace(/\/$/, '')}/${props.id}/`, 301);
